import os
import asyncio
import aiohttp
import jwt
import uuid
import hashlib
import logging
from urllib.parse import urlencode
from dotenv import load_dotenv
from datetime import datetime
import threading
from decimal import Decimal, ROUND_UP, ROUND_DOWN
from logging.handlers import RotatingFileHandler
import signal
import json

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
SEED_AMOUNT = float(os.getenv('SEED_AMOUNT', '50000.0'))  # 단위: KRW
MIN_TRADE_VOLUME = float(os.getenv('MIN_TRADE_VOLUME', '1000'))  # 최소 거래 수량
TARGET_PROFIT_RATE = float(os.getenv('TARGET_PROFIT_RATE', '0.003'))  # 목표 수익률 (0.3%)
ORDER_REFRESH_INTERVAL = int(os.getenv('ORDER_REFRESH_INTERVAL', '30'))  # 주문 갱신 주기 (초)
MAX_OPEN_ORDERS = int(os.getenv('MAX_OPEN_ORDERS', '10'))  # 최대 미체결 주문 수

# 전역 변수
shutdown_event = threading.Event()
open_orders = {}
orders_lock = threading.Lock()
total_profit = 0.0  # 누적 수익

# 이벤트 루프 설정
loop = asyncio.get_event_loop()

def handle_shutdown(signum, frame):
    logger.info("프로그램을 종료합니다...")
    shutdown_event.set()
    loop.stop()

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

async def make_api_request(method, endpoint, params=None, data=None):
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

    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            if method == 'GET':
                async with session.get(url, params=params, headers=headers) as response:
                    response.raise_for_status()
                    return await response.json()
            elif method == 'POST':
                async with session.post(url, json=data, headers=headers) as response:
                    response.raise_for_status()
                    return await response.json()
            elif method == 'DELETE':
                async with session.delete(url, params=params, headers=headers) as response:
                    response.raise_for_status()
                    return await response.json()
            else:
                logger.error(f"지원하지 않는 메서드: {method}")
                return None
        except aiohttp.ClientError as e:
            logger.error(f"API 요청 오류: {e} | URL: {url} | 메서드: {method}")
            return None

async def get_account_balance(currency):
    endpoint = '/v1/accounts'
    response = await make_api_request('GET', endpoint)
    if response:
        for item in response:
            if item['currency'] == currency:
                return float(item['balance'])
    return 0.0

def get_tick_size(price):
    if price >= 2000000:
        return 1000
    elif price >= 1000000:
        return 500
    elif price >= 500000:
        return 100
    elif price >= 100000:
        return 50
    elif price >= 10000:
        return 10
    elif price >= 1000:
        return 1
    elif price >= 100:
        return 0.1
    elif price >= 10:
        return 0.01
    elif price >= 1:
        return 0.001
    elif price >= 0.1:
        return 0.0001
    elif price >= 0.01:
        return 0.00001
    elif price >= 0.001:
        return 0.000001
    elif price >= 0.0001:
        return 0.0000001
    else:
        return 0.00000001

def round_price(price, tick_size, direction=ROUND_DOWN):
    price_decimal = Decimal(str(price))
    tick_size_decimal = Decimal(str(tick_size))
    if direction == ROUND_DOWN:
        rounded = (price_decimal / tick_size_decimal).to_integral_value(rounding=ROUND_DOWN) * tick_size_decimal
    else:
        rounded = (price_decimal / tick_size_decimal).to_integral_value(rounding=ROUND_UP) * tick_size_decimal
    return float(rounded)

async def place_order(side, volume, price):
    data = {
        'market': MARKET,
        'side': side,
        'volume': str(volume),
        'price': str(price),
        'ord_type': 'limit'
    }
    response = await make_api_request('POST', '/v1/orders', data=data)
    if response and 'uuid' in response:
        logger.info(f"주문 생성 성공: {side} {volume}개 @ {price}원")
        return response['uuid']
    else:
        logger.error(f"주문 생성 실패: {response}")
        return None

async def cancel_order(order_id):
    params = {'uuid': order_id}
    response = await make_api_request('DELETE', '/v1/order', params=params)
    if response and 'uuid' in response:
        logger.info(f"주문 취소 성공: 주문 ID {order_id}")
        return True
    else:
        logger.error(f"주문 취소 실패: {response}")
        return False

async def check_order_status(order_id):
    params = {'uuid': order_id}
    response = await make_api_request('GET', '/v1/order', params=params)
    return response

# 오더북 조회 함수
async def get_orderbook():
    endpoint = '/v1/orderbook'
    params = {'markets': MARKET}
    response = await make_api_request('GET', endpoint, params=params)
    if response:
        return response[0]
    else:
        logger.error("호가 정보를 가져오지 못했습니다.")
        return None

# 주문 배치 함수
async def place_orders():
    while not shutdown_event.is_set():
        try:
            # 미체결 주문 수 확인
            with orders_lock:
                if len(open_orders) >= MAX_OPEN_ORDERS:
                    await asyncio.sleep(1)
                    continue

            orderbook = await get_orderbook()
            if not orderbook:
                await asyncio.sleep(0.1)
                continue

            tick_size = get_tick_size(orderbook['orderbook_units'][0]['ask_price'])

            # 매수 주문 가격 계산
            max_bid_unit = max(orderbook['orderbook_units'], key=lambda x: x['bid_size'])
            max_bid_price = max_bid_unit['bid_price']
            buy_price = round_price(max_bid_price + tick_size, tick_size, direction=ROUND_UP)

            # 매도 주문 가격 계산
            max_ask_unit = max(orderbook['orderbook_units'], key=lambda x: x['ask_size'])
            max_ask_price = max_ask_unit['ask_price']
            sell_price = round_price(max_ask_price - tick_size, tick_size, direction=ROUND_DOWN)

            # 매수/매도 가격 차이 확인 (최소 0.3%)
            price_gap = ((sell_price - buy_price) / buy_price)
            if price_gap < TARGET_PROFIT_RATE:
                logger.info("매수와 매도 가격의 차이가 목표 수익률 미만으로 거래를 진행하지 않습니다.")
                await asyncio.sleep(1)
                continue

            krw_balance = await get_account_balance('KRW')
            shib_balance = await get_account_balance('SHIB')

            # 슬리피지 방지를 위한 주문량 조절
            max_trade_volume = min(MIN_TRADE_VOLUME, krw_balance / buy_price)

            if max_trade_volume < MIN_TRADE_VOLUME:
                logger.info("잔액이 부족하여 거래를 진행하지 않습니다.")
                await asyncio.sleep(1)
                continue

            # 매수 주문 배치
            order_id = await place_order('bid', max_trade_volume, buy_price)
            if order_id:
                with orders_lock:
                    open_orders[order_id] = {
                        'side': 'buy',
                        'price': buy_price,
                        'volume': max_trade_volume,
                        'sell_price': sell_price,  # 미리 계산한 매도 가격 저장
                        'timestamp': datetime.utcnow()
                    }

            await asyncio.sleep(0.1)
        except Exception as e:
            logger.exception(f"주문 배치 중 오류 발생: {e}")
            await asyncio.sleep(0.1)

# 주문 모니터링 함수
async def monitor_orders():
    global total_profit
    while not shutdown_event.is_set():
        try:
            with orders_lock:
                tasks = []
                order_ids = list(open_orders.keys())
                for order_id in order_ids:
                    tasks.append(check_order_status(order_id))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"주문 상태 확인 중 오류 발생: {result}")
                    continue

                order_id = order_ids[i]
                order_info = open_orders[order_id]
                order_status = result

                if order_status and order_status['state'] == 'done':
                    logger.info(f"주문 체결 완료: 주문 ID {order_id}")
                    # 체결된 주문에 따라 대응 주문 생성
                    if order_info['side'] == 'buy':
                        # 미리 계산한 매도 가격으로 매도 주문 생성
                        sell_price = order_info['sell_price']
                        tick_size = get_tick_size(sell_price)
                        sell_price = round_price(sell_price, tick_size, direction=ROUND_DOWN)
                        sell_volume = order_info['volume']
                        new_order_id = await place_order('ask', sell_volume, sell_price)
                        if new_order_id:
                            with orders_lock:
                                open_orders[new_order_id] = {
                                    'side': 'sell',
                                    'price': sell_price,
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
                        logger.info(f"거래 완료: 수익 {profit:.8f}원, 수익률 {profit_rate:.4f}%, 누적 수익 {total_profit:.8f}원")
                    with orders_lock:
                        del open_orders[order_id]
                elif order_status and order_status['state'] == 'cancel':
                    logger.info(f"주문 취소됨: 주문 ID {order_id}")
                    with orders_lock:
                        del open_orders[order_id]
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.exception(f"주문 모니터링 중 오류 발생: {e}")
            await asyncio.sleep(0.1)

# 오래된 주문 취소 함수
async def cancel_old_orders():
    while not shutdown_event.is_set():
        try:
            with orders_lock:
                current_time = datetime.utcnow()
                tasks = []
                order_ids = list(open_orders.keys())
                for order_id in order_ids:
                    order_info = open_orders[order_id]
                    elapsed_time = (current_time - order_info['timestamp']).total_seconds()
                    if elapsed_time > ORDER_REFRESH_INTERVAL:
                        tasks.append(cancel_order(order_id))
                        del open_orders[order_id]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.exception(f"오래된 주문 취소 중 오류 발생: {e}")
            await asyncio.sleep(0.1)

async def main():
    logger.info("트레이딩 봇 시작")

    place_task = asyncio.create_task(place_orders())
    monitor_task = asyncio.create_task(monitor_orders())
    cancel_task = asyncio.create_task(cancel_old_orders())

    await asyncio.gather(place_task, monitor_task, cancel_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("프로그램을 종료합니다.")
