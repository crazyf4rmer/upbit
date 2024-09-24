import os
import requests
import jwt
import uuid
import hashlib
import time
import logging
import json
import threading
from urllib.parse import urlencode, unquote
from dotenv import load_dotenv


# .env 파일 로드 (스크립트 파일의 동일 디렉토리에 위치)
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

# Upbit API Keys 설정 (환경 변수에서 불러오기)
access_key = os.getenv('UPBIT_OPEN_API_ACCESS_KEY')
secret_key = os.getenv('UPBIT_OPEN_API_SECRET_KEY')
server_url = os.getenv('UPBIT_OPEN_API_SERVER_URL', 'https://api.upbit.com')

if not access_key:
    raise ValueError("Upbit access key not found. Please set 'UPBIT_OPEN_API_ACCESS_KEY' in your .env file.")
if not secret_key:
    raise ValueError("Upbit secret key not found. Please set 'UPBIT_OPEN_API_SECRET_KEY' in your .env file.")

# 로깅 설정
logger = logging.getLogger('trading_bot')
logger.setLevel(logging.DEBUG)

# 파일 핸들러 (DEBUG 이상)
file_handler = logging.FileHandler('trading_bot.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 콘솔 핸들러 (INFO 이상)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# 틱 사이즈 계산 함수
def get_tick_size(price):
    # 가격에 따른 틱 사이즈를 동적으로 지정할 수 있습니다.
    if price < 1:
        return 0.0001
    elif price < 10:
        return 0.001
    elif price < 100:
        return 0.01
    elif price < 1000:
        return 0.1
    else:
        return 1.0

# 수익성 검증 함수
def is_profit_possible(buy_price, sell_price, fee_rate=0.001):
    """
    수수료를 고려하여 매도 가격이 수익을 낼 수 있는지 확인합니다.
    fee_rate: 매수 + 매도 수수료 (0.1% = 0.001)
    """
    total_fee_buy = buy_price * fee_rate  # 매수 수수료
    total_fee_sell = sell_price * fee_rate  # 매도 수수료
    total_cost = buy_price + total_fee_buy
    total_revenue = sell_price - total_fee_sell
    profit = total_revenue - total_cost
    logger.debug(f"Buy Price: {buy_price}, Sell Price: {sell_price}, Profit: {profit}")
    return profit > 0

# 주문 상태 추적을 위한 전역 변수 및 락
order_lock = threading.Lock()
active_buy_order_id = None  # 현재 활성화된 매수 주문의 ID
active_sell_orders = {}  # {buy_order_id: sell_order_id}

# 매수/매도 주문 실행 함수
def place_order(market, side, volume, price, linked_order_id=None, ord_type='limit'):
    params = {
        'market': market,
        'side': side,
        'volume': str(volume),  # 문자열로 변환
        'price': str(price),    # 문자열로 변환
        'ord_type': ord_type,
    }
    
    if linked_order_id:
        params['linked_order_id'] = linked_order_id  # 매도 주문이 매수 주문과 연결되었음을 표시

    # URL 인코딩된 쿼리 문자열 생성 (unquote 사용)
    query_string = unquote(urlencode(params, doseq=True)).encode("utf-8")

    # SHA512 해시 생성
    m = hashlib.sha512()
    m.update(query_string)
    query_hash = m.hexdigest()

    # JWT 페이로드 구성
    payload = {
        'access_key': access_key,
        'nonce': str(uuid.uuid4()),
        'query_hash': query_hash,
        'query_hash_alg': 'SHA512',
    }

    # JWT 토큰 생성
    try:
        jwt_token = jwt.encode(payload, secret_key, algorithm='HS512')
        if isinstance(jwt_token, bytes):
            jwt_token = jwt_token.decode('utf-8')
    except Exception as e:
        logger.debug(f"Error encoding JWT: {e}")
        return None

    authorization = f'Bearer {jwt_token}'
    headers = {
        "Authorization": authorization
    }

    # POST 요청 전송 (json=params으로 전송)
    try:
        response = requests.post(server_url + '/v1/orders', json=params, headers=headers)
        response.raise_for_status()

        data = response.json()
        order_id = data.get('uuid')
        if order_id:
            with order_lock:
                if side == 'bid':
                    global active_buy_order_id
                    active_buy_order_id = order_id
                elif side == 'ask':
                    if linked_order_id:
                        active_sell_orders[linked_order_id] = order_id
            return data
        return data
    except requests.exceptions.HTTPError as http_err:
        logger.debug(f"HTTP error occurred: {http_err} - Response: {response.text}")
    except Exception as e:
        logger.debug(f"Error placing order: {e}")
        logger.debug(f"Query string: {query_string}")
        logger.debug(f"Query hash: {query_hash}")

    return None

# 주문 상태 확인 함수
def check_order_status(order_id, side):
    """
    특정 주문의 상태를 확인하는 함수
    """
    url = f"{server_url}/v1/order"
    query = {
        'uuid': order_id
    }
    query_string = urlencode(query).encode("utf-8")

    m = hashlib.sha512()
    m.update(query_string)
    query_hash = m.hexdigest()

    payload = {
        'access_key': access_key,
        'nonce': str(uuid.uuid4()),
        'query_hash': query_hash,
        'query_hash_alg': 'SHA512',
    }

    try:
        jwt_token = jwt.encode(payload, secret_key, algorithm='HS512')
        if isinstance(jwt_token, bytes):
            jwt_token = jwt_token.decode('utf-8')
    except Exception as e:
        logger.debug(f"Error encoding JWT for order status: {e}")
        return None

    authorization = f'Bearer {jwt_token}'
    headers = {
        "Authorization": authorization
    }

    try:
        response = requests.get(url, params=query, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.HTTPError as http_err:
        logger.debug(f"HTTP error occurred while checking order status: {http_err} - Response: {response.text}")
    except Exception as e:
        logger.debug(f"Error checking order status: {e}")

    return None

# 주문 취소 함수
def cancel_order(order_id):
    """
    특정 주문을 취소하는 함수
    """
    url = f"{server_url}/v1/order"
    query = {
        'uuid': order_id
    }
    query_string = urlencode(query).encode("utf-8")

    m = hashlib.sha512()
    m.update(query_string)
    query_hash = m.hexdigest()

    payload = {
        'access_key': access_key,
        'nonce': str(uuid.uuid4()),
        'query_hash': query_hash,
        'query_hash_alg': 'SHA512',
    }

    try:
        jwt_token = jwt.encode(payload, secret_key, algorithm='HS512')
        if isinstance(jwt_token, bytes):
            jwt_token = jwt_token.decode('utf-8')
    except Exception as e:
        logger.debug(f"Error encoding JWT for canceling order: {e}")
        return None

    authorization = f'Bearer {jwt_token}'
    headers = {
        "Authorization": authorization
    }

    try:
        response = requests.delete(url, params=query, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.HTTPError as http_err:
        logger.debug(f"HTTP error occurred while canceling order: {http_err} - Response: {response.text}")
    except Exception as e:
        logger.debug(f"Error canceling order: {e}")
        logger.debug(f"Query string: {query_string}")
        logger.debug(f"Query hash: {query_hash}")

    return None

# 주문서 데이터 처리 함수
def process_orderbook(orderbook):
    try:
        orderbook_units = orderbook['orderbook_units']
        bids = []
        asks = []
        for unit in orderbook_units:
            bids.append({'price': float(unit['bid_price']), 'size': float(unit['bid_size'])})
            asks.append({'price': float(unit['ask_price']), 'size': float(unit['ask_size'])})
        return bids, asks
    except (KeyError, TypeError, IndexError, ValueError) as e:
        logger.debug(f"Error processing orderbook data: {e}")
        return None, None

# 동적 볼륨 임계값 계산 함수 (필요 시 사용)
def get_dynamic_volume_threshold(market='KRW-TRX', period='day'):
    url = "https://api.upbit.com/v1/trades/ticks"
    params = {
        'market': market,
        'count': 200  # 최근 200개의 거래 데이터를 가져옵니다.
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        trades = response.json()
        if isinstance(trades, list):
            total_volume = sum([
                float(trade['trade_price']) * float(trade['trade_volume'])
                for trade in trades
                if 'trade_price' in trade and 'trade_volume' in trade
            ])
            avg_volume = total_volume / len(trades) if trades else 0
            dynamic_threshold = avg_volume * 2  # 평균의 2배를 임계값으로 설정
            return dynamic_threshold
        else:
            logger.debug(f"Unexpected trades format: {trades}")
            return 100000000  # 기본값 설정 (예: 1억 원)
    except Exception as e:
        logger.debug(f"Error fetching dynamic volume threshold: {e}")
        return 100000000  # 기본값 설정 (예: 1억 원)

# 실시간 주문서 가져오기 함수
def fetch_orderbook(market):
    url = f"{server_url}/v1/orderbook"
    params = {'markets': market}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        orderbook_data = response.json()
        return orderbook_data[0] if orderbook_data else None
    except Exception as e:
        logger.debug(f"Error fetching orderbook: {e}")
        return None

# 매수 주문 실행 함수
def execute_buy_order():
    market = 'KRW-TRX'  # TRX 시장 심볼로 변경
    with order_lock:
        # 활성화된 매수 주문이 있는지 확인
        if active_buy_order_id is not None:
            logger.debug("이미 활성화된 매수 주문이 있어 새로운 매수 주문을 실행하지 않습니다.")
            return

    orderbook = fetch_orderbook(market)
    if not orderbook:
        logger.debug("Failed to fetch orderbook.")
        return

    bids, asks = process_orderbook(orderbook)
    if not bids or not asks:
        logger.debug("Failed to process orderbook.")
        return

    best_bid = bids[0]['price']
    best_ask = asks[0]['price']
    mid_price = (best_bid + best_ask) / 2

    # 매수 가격 설정 (스프레드 최소 0.3원 또는 TRX의 가격대에 맞게 조정)
    tick_size = get_tick_size(mid_price)
    buy_price = round(mid_price - 0.3, len(str(tick_size).split('.')[-1]))
    logger.debug(f"Calculated buy price: {buy_price} with tick size: {tick_size}")

    # 주문 수량 설정 (예: TRX의 가격대에 맞게 조정)
    volume = '150'  # TRX 가격에 따라 적절히 조정 (예: 150 TRX)

    # 주문 실행
    response_buy = place_order(market, 'bid', volume, buy_price)

    current_time = time.strftime("%H:%M:%S", time.localtime())

    if response_buy and 'uuid' in response_buy:
        logger.info(f"{current_time} - 매수 주문: {buy_price}원, 수량: {volume} TRX")
    else:
        logger.info(f"{current_time} - 매수 주문 실행 실패")
        if response_buy and 'uuid' not in response_buy:
            logger.info(f"{current_time} - 매수 주문 실패")

# 매수 주문 체결 시 즉시 매도 주문을 실행하는 함수
def place_sell_order_on_buy(order_id, buy_price, volume):
    fee_rate = 0.001  # 0.1%
    target_profit_rate = 0.002  # 0.2%
    
    # 수수료를 고려한 매도 가격 설정
    sell_price = buy_price * (1 + target_profit_rate + fee_rate)
    tick_size = get_tick_size(sell_price)
    sell_price = round(sell_price / tick_size) * tick_size  # 정확한 tick size로 반올림

    logger.debug(f"Setting sell price to {sell_price} based on buy price {buy_price} with tick size {tick_size}")

    # 수익성 검증
    if not is_profit_possible(buy_price, sell_price):
        logger.debug(f"Immediate sell profit not possible: Buy at {buy_price}, Sell at {sell_price}")
        return

    response_sell = place_order('KRW-TRX', 'ask', volume, sell_price, linked_order_id=order_id)

    current_time = time.strftime("%H:%M:%S", time.localtime())

    if response_sell and 'uuid' in response_sell:
        sell_order_id = response_sell['uuid']
        with order_lock:
            active_sell_orders[order_id] = sell_order_id
        logger.info(f"{current_time} - 매도 주문 실행: {sell_price}원, 수량: {volume} TRX (매수 주문 ID: {order_id})")
    else:
        logger.info(f"{current_time} - 매도 주문 실행 실패: {sell_price}원, 수량: {volume} TRX (매수 주문 ID: {order_id})")

# 주문 상태 모니터링 및 관리 함수
def monitor_orders():
    global active_buy_order_id, active_sell_orders
    while True:
        time.sleep(5)  # 5초마다 주문 상태 확인
        with order_lock:
            # 매수 주문 상태 확인
            if active_buy_order_id:
                status = check_order_status(active_buy_order_id, 'bid')
                if status:
                    state = status.get('state')
                    current_time = time.strftime("%H:%M:%S", time.localtime())
                    if state == 'done':
                        buy_price = float(status.get('price'))
                        buy_volume = float(status.get('volume'))
                        logger.info(f"{current_time} - 매수 주문 완료: {buy_price}원, 주문 ID: {active_buy_order_id}")
                        # 매수 주문이 체결되면 즉시 매도 주문 실행
                        place_sell_order_on_buy(active_buy_order_id, buy_price, buy_volume)
                        # 매수 주문 ID 초기화
                        active_buy_order_id = None
                    elif state in ['cancelled', 'failed']:
                        logger.info(f"{current_time} - 매수 주문 취소됨: {status.get('price')}원, 주문 ID: {active_buy_order_id}")
                        active_buy_order_id = None

            # 매도 주문 상태 확인
            for buy_id, sell_id in list(active_sell_orders.items()):
                status = check_order_status(sell_id, 'ask')
                if status:
                    state = status.get('state')
                    current_time = time.strftime("%H:%M:%S", time.localtime())
                    if state == 'done':
                        logger.info(f"{current_time} - 매도 주문 완료: {status.get('price')}원, 주문 ID: {sell_id}")
                        del active_sell_orders[buy_id]
                    elif state in ['cancelled', 'failed']:
                        logger.info(f"{current_time} - 매도 주문 취소됨: {status.get('price')}원, 주문 ID: {sell_id}")
                        del active_sell_orders[buy_id]

            # 활성화된 매수 주문 외에 다른 주문이 있는지 확인하고, 가격이 벗어나면 취소
            # 현재 구조에서는 매수 주문과 매도 주문만 추적하므로, 추가적인 취소 로직은 필요하지 않습니다.

# 주문 스케줄러 함수
def schedule_orders():
    while True:
        current_time = time.time()
        # 다음 1분 단위까지 대기
        sleep_time = 60 - (current_time % 60)
        logger.debug(f"Waiting for {sleep_time:.2f} seconds until next order execution.")
        time.sleep(sleep_time)
        execute_buy_order()

if __name__ == "__main__":
    # 주문 상태 모니터링 스레드 시작
    monitor_thread = threading.Thread(target=monitor_orders)
    monitor_thread.daemon = True
    monitor_thread.start()
    logger.debug("Order monitoring thread started.")

    # 주문 스케줄러 스레드 시작
    schedule_thread = threading.Thread(target=schedule_orders)
    schedule_thread.daemon = True
    schedule_thread.start()
    logger.debug("Order scheduling thread started.")

    logger.info("Trading bot is running with time-based grid strategy for TRX. Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")