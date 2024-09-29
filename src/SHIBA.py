import os
import requests
import jwt
import uuid
import hashlib
import time
import logging
from urllib.parse import urlencode
from dotenv import load_dotenv
from datetime import datetime
import threading
from decimal import Decimal, ROUND_UP
from logging.handlers import RotatingFileHandler
import signal
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json  # JSON 모듈 임포트 추가

# .env 파일 로드
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

# Upbit API Keys 설정
ACCESS_KEY = os.getenv('UPBIT_OPEN_API_ACCESS_KEY')
SECRET_KEY = os.getenv('UPBIT_OPEN_API_SECRET_KEY')
SERVER_URL = os.getenv('UPBIT_OPEN_API_SERVER_URL', 'https://api.upbit.com')

if not ACCESS_KEY or not SECRET_KEY:
    raise ValueError("Upbit API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 로깅 설정
logger = logging.getLogger('trading_bot')
logger.setLevel(logging.INFO)

# 파일 핸들러
file_handler = RotatingFileHandler('trading_bot.log', maxBytes=10*1024*1024, backupCount=5)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 콘솔 핸들러
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# 트레이딩 설정
MARKET = 'KRW-SHIB'
COIN_SYMBOL = 'SHIB'
SEED_AMOUNT = float(os.getenv('SEED_AMOUNT', '200000.0'))  # 단위: KRW
MIN_TRADE_VOLUME = float(os.getenv('MIN_TRADE_VOLUME', '1000'))  # 최소 거래 수량
TARGET_PROFIT_RATE = float(os.getenv('TARGET_PROFIT_RATE', '0.0025'))  # 목표 수익률 (0.25%)
ORDER_REFRESH_INTERVAL = int(os.getenv('ORDER_REFRESH_INTERVAL', '30'))  # 주문 갱신 주기 (초)
MAX_OPEN_ORDERS = int(os.getenv('MAX_OPEN_ORDERS', '10'))  # 최대 미체결 주문 수

# 세션 설정
session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST", "DELETE"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

# 전역 변수
shutdown_event = threading.Event()
open_orders = {}
orders_lock = threading.Lock()
total_profit = 0.0  # 누적 수익

def handle_shutdown(signum, frame):
    logger.info("프로그램을 종료합니다...")
    shutdown_event.set()

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

# API 함수들
def get_jwt_token(query=None):
    payload = {
        'access_key': ACCESS_KEY,
        'nonce': str(uuid.uuid4()),
    }
    if query:
        if isinstance(query, dict):
            # POST 요청의 경우 JSON 직렬화된 문자열 사용
            query_string = json.dumps(query, separators=(',', ':'), ensure_ascii=False)
        else:
            # GET 요청의 경우 URL 인코딩된 쿼리 문자열 사용
            query_string = urlencode(query)
        m = hashlib.sha512()
        m.update(query_string.encode('utf-8'))
        query_hash = m.hexdigest()
        payload['query_hash'] = query_hash
        payload['query_hash_alg'] = 'SHA512'
    try:
        jwt_token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
        if isinstance(jwt_token, bytes):
            jwt_token = jwt_token.decode('utf-8')
        return jwt_token
    except Exception as e:
        logger.exception(f"JWT 토큰 생성 오류: {e}")
        return None

def make_api_request(method, endpoint, params=None, data=None):
    url = f"{SERVER_URL}{endpoint}"
    jwt_token = get_jwt_token(data if method == 'POST' else params)
    if jwt_token is None:
        logger.error("JWT 토큰 생성 실패로 요청을 중단합니다.")
        return None
    headers = {
        "Authorization": f"Bearer {jwt_token}"
    }
    if method == 'POST':
        headers['Content-Type'] = 'application/json'
    try:
        if method == 'GET':
            response = session.get(url, params=params, headers=headers, timeout=10)
        elif method == 'POST':
            response = session.post(url, json=data, headers=headers, timeout=10)
        elif method == 'DELETE':
            response = session.delete(url, params=params, headers=headers, timeout=10)
        else:
            logger.error(f"지원하지 않는 메서드: {method}")
            return None
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API 요청 오류: {e} | URL: {url} | 메서드: {method}")
        return None

def get_account_balance(currency):
    endpoint = '/v1/accounts'
    response = make_api_request('GET', endpoint)
    if response:
        for item in response:
            if item['currency'] == currency:
                return float(item['balance'])
    return 0.0

def get_orderbook():
    endpoint = '/v1/orderbook'
    params = {'markets': MARKET}
    response = make_api_request('GET', endpoint, params=params)
    if response:
        return response[0]
    else:
        logger.error("호가 정보를 가져오지 못했습니다.")
        return None

def get_tick_size(price):
    if price < 0.1:
        return 0.0001
    elif price < 1:
        return 0.001
    elif price < 10:
        return 0.01
    elif price < 100:
        return 0.1
    elif price < 1000:
        return 1
    elif price < 10000:
        return 5
    elif price < 100000:
        return 10
    else:
        return 50

def round_price(price, tick_size):
    price_decimal = Decimal(str(price))
    tick_size_decimal = Decimal(str(tick_size))
    rounded = (price_decimal / tick_size_decimal).to_integral_value(rounding=ROUND_UP) * tick_size_decimal
    return float(rounded)

def place_order(side, volume, price):
    data = {
        'market': MARKET,
        'side': side,
        'volume': str(volume),
        'price': str(price),
        'ord_type': 'limit'
    }
    response = make_api_request('POST', '/v1/orders', data=data)
    if response and 'uuid' in response:
        logger.info(f"주문 생성 성공: {side} {volume}개 @ {price}원")
        return response['uuid']
    else:
        logger.error(f"주문 생성 실패: {response}")
        return None

def cancel_order(order_id):
    params = {'uuid': order_id}
    response = make_api_request('DELETE', '/v1/order', params=params)
    if response and 'uuid' in response:
        logger.info(f"주문 취소 성공: 주문 ID {order_id}")
        return True
    else:
        logger.error(f"주문 취소 실패: {response}")
        return False

def get_open_orders():
    params = {'market': MARKET, 'state': 'wait'}
    response = make_api_request('GET', '/v1/orders', params=params)
    if response:
        return response
    else:
        logger.error("미체결 주문을 가져오지 못했습니다.")
        return []

def check_order_status(order_id):
    params = {'uuid': order_id}
    response = make_api_request('GET', '/v1/order', params=params)
    return response

# 주문 배치 함수
def place_orders():
    while not shutdown_event.is_set():
        try:
            orderbook = get_orderbook()
            if not orderbook:
                time.sleep(1)
                continue

            bids = orderbook['orderbook_units'][0]['bid_price']  # 매수 호가
            asks = orderbook['orderbook_units'][0]['ask_price']  # 매도 호가

            tick_size = get_tick_size(bids)

            # 매수 주문 가격: 매수 호가 한두 틱 아래
            buy_price = bids - tick_size * 1
            buy_price = round_price(buy_price, tick_size)

            # 매도 주문 가격: 매도 호가 한두 틱 위
            sell_price = asks + tick_size * 1
            sell_price = round_price(sell_price, tick_size)

            krw_balance = get_account_balance('KRW')
            shib_balance = get_account_balance('SHIB')

            # 매수 주문 배치
            if krw_balance >= MIN_TRADE_VOLUME * buy_price:
                buy_volume = MIN_TRADE_VOLUME
                order_id = place_order('bid', buy_volume, buy_price)
                if order_id:
                    with orders_lock:
                        open_orders[order_id] = {
                            'side': 'buy',
                            'price': buy_price,
                            'volume': buy_volume,
                            'timestamp': datetime.utcnow()
                        }

            # 매도 주문 배치
            if shib_balance >= MIN_TRADE_VOLUME:
                sell_volume = MIN_TRADE_VOLUME
                order_id = place_order('ask', sell_volume, sell_price)
                if order_id:
                    with orders_lock:
                        open_orders[order_id] = {
                            'side': 'sell',
                            'price': sell_price,
                            'volume': sell_volume,
                            'timestamp': datetime.utcnow()
                        }

            time.sleep(1)
        except Exception as e:
            logger.exception(f"주문 배치 중 오류 발생: {e}")
            time.sleep(1)

# 주문 모니터링 함수
def monitor_orders():
    global total_profit
    while not shutdown_event.is_set():
        try:
            with orders_lock:
                for order_id in list(open_orders.keys()):
                    order_info = open_orders[order_id]
                    order_status = check_order_status(order_id)
                    if order_status and order_status['state'] == 'done':
                        logger.info(f"주문 체결 완료: 주문 ID {order_id}")
                        # 체결된 주문에 따라 대응 주문 생성
                        if order_info['side'] == 'buy':
                            # 매수 주문이 체결되었으므로 목표 수익률로 매도 주문 생성
                            target_price = order_info['price'] * (1 + TARGET_PROFIT_RATE)
                            tick_size = get_tick_size(target_price)
                            target_price = round_price(target_price, tick_size)
                            sell_volume = order_info['volume']
                            new_order_id = place_order('ask', sell_volume, target_price)
                            if new_order_id:
                                open_orders[new_order_id] = {
                                    'side': 'sell',
                                    'price': target_price,
                                    'volume': sell_volume,
                                    'buy_price': order_info['price'],
                                    'timestamp': datetime.utcnow()
                                }
                        elif order_info['side'] == 'sell':
                            # 매도 주문이 체결되었으므로 수익 계산
                            buy_price = order_info.get('buy_price', 0)
                            sell_price = order_info['price']
                            volume = order_info['volume']
                            fee_rate = 0.0005  # 업비트 거래 수수료 0.05%
                            # 총 매수 금액 및 수수료
                            total_buy = buy_price * volume
                            total_buy_fee = total_buy * fee_rate
                            # 총 매도 금액 및 수수료
                            total_sell = sell_price * volume
                            total_sell_fee = total_sell * fee_rate
                            # 실제 수익 계산
                            profit = total_sell - total_sell_fee - total_buy - total_buy_fee
                            total_profit += profit
                            profit_rate = (profit / total_buy) * 100  # 수익률 (%)
                            logger.info(f"거래 완료: 수익 {profit:.2f}원, 수익률 {profit_rate:.4f}%, 누적 수익 {total_profit:.2f}원")
                        del open_orders[order_id]
                    elif order_status and order_status['state'] == 'cancel':
                        logger.info(f"주문 취소됨: 주문 ID {order_id}")
                        del open_orders[order_id]
            time.sleep(1)
        except Exception as e:
            logger.exception(f"주문 모니터링 중 오류 발생: {e}")
            time.sleep(1)

# 오래된 주문 취소 함수
def cancel_old_orders():
    while not shutdown_event.is_set():
        try:
            with orders_lock:
                current_time = datetime.utcnow()
                for order_id in list(open_orders.keys()):
                    order_info = open_orders[order_id]
                    order_status = check_order_status(order_id)
                    if order_status and order_status['state'] == 'wait':
                        elapsed_time = (current_time - order_info['timestamp']).total_seconds()
                        if elapsed_time > ORDER_REFRESH_INTERVAL:
                            cancel_order(order_id)
                            del open_orders[order_id]
            time.sleep(1)
        except Exception as e:
            logger.exception(f"오래된 주문 취소 중 오류 발생: {e}")
            time.sleep(1)

def main():
    logger.info("트레이딩 봇 시작")

    place_thread = threading.Thread(target=place_orders)
    monitor_thread = threading.Thread(target=monitor_orders)
    cancel_thread = threading.Thread(target=cancel_old_orders)

    place_thread.start()
    monitor_thread.start()
    cancel_thread.start()

    while not shutdown_event.is_set():
        time.sleep(1)

    place_thread.join()
    monitor_thread.join()
    cancel_thread.join()

    logger.info("트레이딩 봇 종료")

if __name__ == "__main__":
    main()
