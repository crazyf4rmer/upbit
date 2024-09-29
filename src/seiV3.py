import os
import requests
import jwt
import uuid
import hashlib
import time
import logging
from urllib.parse import urlencode
from dotenv import load_dotenv
from datetime import datetime, timedelta
import threading
from decimal import Decimal, ROUND_UP
from logging.handlers import RotatingFileHandler
import signal
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np

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
MIN_BUY_AMOUNT_KRW = float(os.getenv('MIN_BUY_AMOUNT_KRW', '5000'))  # 최소 매수 금액
MIN_SELL_VOLUME = float(os.getenv('MIN_SELL_VOLUME', '1000'))  # 최소 매도 수량
TARGET_PROFIT_RATE = float(os.getenv('TARGET_PROFIT_RATE', '0.005'))  # 목표 수익률 (0.5%)
ATR_PERIOD = int(os.getenv('ATR_PERIOD', '14'))  # ATR 계산 기간

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
position = None  # 현재 포지션 ('long' 또는 None)
entry_price = 0.0  # 진입 가격

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
        query_string = urlencode(query).encode('utf-8')
        m = hashlib.sha512()
        m.update(query_string)
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
    jwt_token = get_jwt_token(params if method == 'GET' else data)
    if jwt_token is None:
        logger.error("JWT 토큰 생성 실패로 요청을 중단합니다.")
        return None
    headers = {
        "Authorization": f"Bearer {jwt_token}"
    }
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

def get_current_price():
    endpoint = '/v1/ticker'
    params = {'markets': MARKET}
    response = make_api_request('GET', endpoint, params=params)
    if response:
        return float(response[0]['trade_price'])
    else:
        logger.error("현재 가격을 가져오지 못했습니다.")
        return None

def get_account_balance(currency):
    endpoint = '/v1/accounts'
    response = make_api_request('GET', endpoint)
    if response:
        for item in response:
            if item['currency'] == currency:
                return float(item['balance'])
    return 0.0

def get_ohlcv(minutes, count):
    endpoint = f'/v1/candles/minutes/{minutes}'
    params = {'market': MARKET, 'count': count}
    response = make_api_request('GET', endpoint, params=params)
    if response:
        return response[::-1]  # 최신순으로 반환되므로 역순으로 정렬
    else:
        logger.error("OHLCV 데이터를 가져오지 못했습니다.")
        return None

def calculate_atr(period):
    ohlcv = get_ohlcv(1, period + 1)
    if not ohlcv or len(ohlcv) < period + 1:
        logger.error("ATR 계산을 위한 충분한 데이터가 없습니다.")
        return None
    tr_list = []
    for i in range(1, len(ohlcv)):
        current_high = ohlcv[i]['high_price']
        current_low = ohlcv[i]['low_price']
        prev_close = ohlcv[i-1]['trade_price']
        tr = max(current_high - current_low, abs(current_high - prev_close), abs(current_low - prev_close))
        tr_list.append(tr)
    atr = sum(tr_list[-period:]) / period
    return atr

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

def place_order(side, volume, price=None):
    data = {
        'market': MARKET,
        'side': side,
        'volume': str(volume),
        'ord_type': 'limit' if price else 'price'
    }
    if price:
        data['price'] = str(price)
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

def check_order_status(order_id):
    params = {'uuid': order_id}
    response = make_api_request('GET', '/v1/order', params=params)
    return response

# 이동 평균 계산 함수
def calculate_moving_average(prices, window):
    if len(prices) < window:
        return None
    return sum(prices[-window:]) / window

# 최근 가격 가져오기 함수
def get_recent_prices(market, count):
    endpoint = '/v1/candles/minutes/1'
    params = {'market': market, 'count': count}
    response = make_api_request('GET', endpoint, params=params)
    if response:
        prices = [float(candle['trade_price']) for candle in reversed(response)]
        return prices
    else:
        logger.error("최근 가격 데이터를 가져오지 못했습니다.")
        return None

# 추세 추종 전략 함수
def trend_following_strategy():
    global position, entry_price
    short_window = 5  # 단기 이동 평균 기간
    long_window = 20  # 장기 이동 평균 기간

    prices = get_recent_prices(MARKET, long_window + 1)
    if not prices:
        return

    short_ma = calculate_moving_average(prices, short_window)
    long_ma = calculate_moving_average(prices, long_window)

    if not short_ma or not long_ma:
        return

    current_price = prices[-1]
    krw_balance = get_account_balance('KRW')
    shib_balance = get_account_balance('SHIB')

    # 골든크로스 발생 (매수 신호)
    if short_ma > long_ma and position != 'long' and krw_balance >= MIN_BUY_AMOUNT_KRW:
        volume = SEED_AMOUNT / current_price
        volume = round(volume, 0)  # 소수점 이하 제거
        order_id = place_order('bid', volume, current_price)
        if order_id:
            position = 'long'
            entry_price = current_price
            logger.info(f"골든크로스 발생으로 매수 주문 체결: {volume}개 @ {current_price}원")

    # 데드크로스 발생 (매도 신호)
    elif short_ma < long_ma and position == 'long' and shib_balance >= MIN_SELL_VOLUME:
        volume = shib_balance
        volume = round(volume, 0)
        order_id = place_order('ask', volume, current_price)
        if order_id:
            position = None
            profit = (current_price - entry_price) * volume
            logger.info(f"데드크로스 발생으로 매도 주문 체결: {volume}개 @ {current_price}원, 수익: {profit}원")

# 전략 선택 함수 (간단히 수정)
def choose_strategy():
    return 'trend_following'

# 메인 루프 수정
def main():
    logger.info("트레이딩 봇 시작")
    while not shutdown_event.is_set():
        try:
            strategy = choose_strategy()
            if strategy == 'trend_following':
                trend_following_strategy()
            else:
                logger.info("관망 중...")
            time.sleep(60)
        except Exception as e:
            logger.exception(f"메인 루프 오류: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
