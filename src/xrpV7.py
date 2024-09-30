import os
import requests
import jwt
import uuid
import hashlib
import math
import time
import logging
from urllib.parse import urlencode, unquote
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import threading
import pandas as pd
from logging.handlers import RotatingFileHandler
from collections import defaultdict

# 환경 변수 및 설정
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

access_key = os.getenv('UPBIT_OPEN_API_ACCESS_KEY')
secret_key = os.getenv('UPBIT_OPEN_API_SECRET_KEY')
server_url = os.getenv('UPBIT_OPEN_API_SERVER_URL', 'https://api.upbit.com')

max_consecutive_failures = 3
consecutive_failures = 0

SEED_AMOUNT = 100000.0  # 초기 시드 금액 (KRW)
MIN_SEED_AMOUNT = 10000.0  # 신규 매수 주문 활성화를 위한 최소 시드 금액 (KRW)
MIN_ORDER_AMOUNT = 5000  # 최소 주문 금액 (KRW)

# 로깅 설정
logger = logging.getLogger('trading_bot')
logger.setLevel(logging.DEBUG)

file_handler = RotatingFileHandler('trading_bot.log', maxBytes=10*1024*1024, backupCount=5)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# 전역 변수 및 락
trade_in_progress = False
trade_lock = threading.Lock()
trading_paused = False
trading_pause_lock = threading.Lock()
available_seed = SEED_AMOUNT
totals_lock = threading.Lock()
file_lock = threading.Lock()
new_buy_orders_allowed = True
new_buy_orders_lock = threading.Lock()

class OrderPair:
    def __init__(self, buy_order):
        self.buy_order = buy_order
        self.sell_order = None
        self.sell_order_time = None

class TradingManager:
    def __init__(self):
        self.order_pairs = {}
        self.reserved_amounts = {}
        self.lock = threading.Lock()

    def add_buy_order(self, buy_order):
        with self.lock:
            self.order_pairs[buy_order['uuid']] = OrderPair(buy_order)
            self.reserved_amounts[buy_order['uuid']] = float(buy_order['price']) * float(buy_order['volume'])

    def update_sell_order(self, buy_order_id, sell_order):
        with self.lock:
            if buy_order_id in self.order_pairs:
                self.order_pairs[buy_order_id].sell_order = sell_order
                self.order_pairs[buy_order_id].sell_order_time = datetime.now()

    def get_order_pair(self, buy_order_id):
        with self.lock:
            return self.order_pairs.get(buy_order_id)

    def remove_order_pair(self, buy_order_id):
        with self.lock:
            if buy_order_id in self.order_pairs:
                del self.order_pairs[buy_order_id]
            if buy_order_id in self.reserved_amounts:
                del self.reserved_amounts[buy_order_id]

    def get_total_reserved_amount(self):
        with self.lock:
            return sum(self.reserved_amounts.values())

    def get_reserved_amount(self, buy_order_id):
        with self.lock:
            return self.reserved_amounts.get(buy_order_id, 0)

    def remove_reserved_amount(self, buy_order_id):
        with self.lock:
            if buy_order_id in self.reserved_amounts:
                del self.reserved_amounts[buy_order_id]

class NonceGenerator:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_nonce = int(time.time() * 1000)
        self.counter = 0

    def get_nonce(self):
        with self.lock:
            current_nonce = int(time.time() * 1000)
            if current_nonce <= self.last_nonce:
                current_nonce = self.last_nonce + 1
            self.last_nonce = current_nonce
            self.counter += 1
            return f"{current_nonce}{self.counter:03d}"

nonce_generator = NonceGenerator()

@contextmanager
def open_order_lock():
    file_lock.acquire()
    try:
        yield
    finally:
        file_lock.release()

def get_tick_size(price):
    if price < 1000:
        return 0.1
    else:
        return 1

def adjust_to_tick_size(price):
    tick_size = get_tick_size(price)
    return math.floor(price / tick_size) * tick_size

def is_profit_possible(buy_price, sell_price, fee_rate=0.0005, min_profit_rate=0.002):
    profit = sell_price - buy_price - (sell_price + buy_price) * fee_rate
    profit_rate = profit / buy_price
    logger.debug(f"Buy Price: {buy_price}, Sell Price: {sell_price}, Profit: {profit}, Profit Rate: {profit_rate:.4f}")
    return profit_rate > min_profit_rate

def get_current_krw_balance():
    url = f"{server_url}/v1/accounts"
    payload = {
        'access_key': access_key,
        'nonce': str(uuid.uuid4()),
    }
    jwt_token = jwt.encode(payload, secret_key, algorithm='HS512')
    authorize_token = 'Bearer {}'.format(jwt_token)
    headers = {"Authorization": authorize_token}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        for account in response.json():
            if account['currency'] == 'KRW':
                return float(account['balance'])
        return 0.0
    except Exception as e:
        logger.error(f"잔고 조회 중 오류 발생: {e}")
        return 0.0

def place_order(market, side, volume, price, ord_type='limit'):
    query = {
        'market': market,
        'side': side,
        'volume': str(volume),
        'price': str(price),
        'ord_type': ord_type,
    }
    query_string = urlencode(query).encode()

    m = hashlib.sha512()
    m.update(query_string)
    query_hash = m.hexdigest()

    payload = {
        'access_key': access_key,
        'nonce': nonce_generator.get_nonce(),
        'query_hash': query_hash,
        'query_hash_alg': 'SHA512',
    }

    jwt_token = jwt.encode(payload, secret_key, algorithm='HS512')
    authorize_token = 'Bearer {}'.format(jwt_token)
    headers = {"Authorization": authorize_token}

    try:
        response = requests.post(f"{server_url}/v1/orders", params=query, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"주문 실행 중 오류 발생: {e}")
        if hasattr(e.response, 'text'):
            logger.error(f"오류 응답: {e.response.text}")
        return None

def check_order_status(order_id):
    query = {
        'uuid': order_id,
    }
    query_string = urlencode(query).encode()

    m = hashlib.sha512()
    m.update(query_string)
    query_hash = m.hexdigest()

    payload = {
        'access_key': access_key,
        'nonce': str(uuid.uuid4()),
        'query_hash': query_hash,
        'query_hash_alg': 'SHA512',
    }

    jwt_token = jwt.encode(payload, secret_key, algorithm='HS512')
    authorize_token = 'Bearer {}'.format(jwt_token)
    headers = {"Authorization": authorize_token}

    try:
        response = requests.get(f"{server_url}/v1/order", params=query, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"주문 상태 확인 중 오류 발생: {e}")
        return None

def cancel_order(order_id, trading_manager):
    order_status = check_order_status(order_id)
    if order_status is None:
        logger.error(f"주문 상태 확인 실패: ID {order_id}")
        return None
    
    if order_status['state'] == 'done':
        logger.info(f"주문 {order_id}는 이미 체결되었습니다. 취소할 필요가 없습니다.")
        return None
    elif order_status['state'] in ['cancel', 'fail']:
        logger.info(f"주문 {order_id}는 이미 취소되었거나 실패했습니다.")
        return None
    elif order_status['state'] in ['wait', 'watch']:
        query = {
            'uuid': order_id,
        }
        query_string = urlencode(query).encode()

        m = hashlib.sha512()
        m.update(query_string)
        query_hash = m.hexdigest()

        payload = {
            'access_key': access_key,
            'nonce': nonce_generator.get_nonce(),
            'query_hash': query_hash,
            'query_hash_alg': 'SHA512',
        }

        jwt_token = jwt.encode(payload, secret_key, algorithm='HS512')
        authorize_token = 'Bearer {}'.format(jwt_token)
        headers = {"Authorization": authorize_token}

        try:
            response = requests.delete(f"{server_url}/v1/order", params=query, headers=headers)
            response.raise_for_status()
            logger.info(f"주문 {order_id} 취소 성공")
            trading_manager.remove_reserved_amount(order_id)
            return response.json()
        except requests.exceptions.RequestException as e:
            if response.status_code == 400 and "Order not found" in response.text:
                logger.warning(f"주문 {order_id}를 찾을 수 없습니다. 이미 체결되었거나 취소되었을 수 있습니다.")
                return None
            else:
                logger.error(f"주문 취소 중 오류 발생: {e}")
                if hasattr(response, 'text'):
                    logger.error(f"응답 내용: {response.text}")
                return None
    else:
        logger.info(f"주문 {order_id}의 상태가 {order_status['state']}입니다. 취소할 수 없습니다.")
        return None

def fetch_orderbook(market):
    url = f"{server_url}/v1/orderbook"
    params = {'markets': market}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()[0]
    except Exception as e:
        logger.error(f"오더북 조회 중 오류 발생: {e}")
        return None

def calculate_optimal_buy_price(orderbook):
    bids = orderbook['orderbook_units']
    volume_at_price = defaultdict(float)
    
    for bid in bids:
        price = float(bid['bid_price'])
        volume = float(bid['bid_size'])
        volume_at_price[price] += volume
    
    sorted_prices = sorted(volume_at_price.keys(), reverse=True)
    
    best_price = sorted_prices[0]
    best_volume = volume_at_price[best_price]
    
    for price in sorted_prices[1:]:
        if volume_at_price[price] > best_volume:
            best_price = price
            best_volume = volume_at_price[price]
    
    optimal_price = best_price + get_tick_size(best_price)
    return adjust_to_tick_size(optimal_price)

def calculate_optimal_sell_price(buy_price, orderbook, min_profit_rate=0.002):
    min_sell_price = buy_price * (1 + min_profit_rate)
    asks = orderbook['orderbook_units']
    volume_at_price = defaultdict(float)
    
    for ask in asks:
        price = float(ask['ask_price'])
        volume = float(ask['ask_size'])
        if price >= min_sell_price:
            volume_at_price[price] += volume
    
    if not volume_at_price:
        return adjust_to_tick_size(min_sell_price)
    
    sorted_prices = sorted(volume_at_price.keys())
    
    best_price = sorted_prices[0]
    best_volume = volume_at_price[best_price]
    
    for price in sorted_prices[1:]:
        if volume_at_price[price] > best_volume:
            best_price = price
            best_volume = volume_at_price[price]
    
    optimal_price = best_price - get_tick_size(best_price)
    return adjust_to_tick_size(max(optimal_price, min_sell_price))

def execute_buy_order(trading_manager):
    global available_seed, new_buy_orders_allowed
    
    with new_buy_orders_lock:
        if not new_buy_orders_allowed:
            logger.info("신규 매수 주문이 비활성화되어 있습니다.")
            return

    orderbook = fetch_orderbook('KRW-XRP')
    if not orderbook:
        logger.error("오더북 조회 실패")
        return

    buy_price = calculate_optimal_buy_price(orderbook)
    
    max_volume = min(8, available_seed / buy_price)
    buy_volume = math.floor(max_volume * 100000) / 100000

    total_buy_amount = buy_price * buy_volume

    if total_buy_amount < MIN_ORDER_AMOUNT:
        logger.warning(f"주문 금액({total_buy_amount:.2f}원)이 최소 주문 금액({MIN_ORDER_AMOUNT}원)보다 작습니다. 주문을 건너뜁니다.")
        return

    if total_buy_amount > available_seed:
        logger.warning(f"주문 금액({total_buy_amount:.2f}원)이 사용 가능한 시드 머니({available_seed:.2f}원)를 초과합니다. 주문을 건너뜁니다.")
        return

    response_buy = place_order('KRW-XRP', 'bid', buy_volume, buy_price)
    if response_buy and 'uuid' in response_buy:
        buy_order = {'uuid': response_buy['uuid'], 'price': buy_price, 'volume': buy_volume}
        trading_manager.add_buy_order(buy_order)
        
        with totals_lock:
            available_seed -= total_buy_amount
        
        logger.info(f"매수 주문 생성됨: 주문 ID {buy_order['uuid']}, 가격: {buy_price:.2f}원, 수량: {buy_volume} XRP, 총 금액: {total_buy_amount:.2f}원")
    else:
        logger.error(f"매수 주문 생성 실패: 가격 {buy_price:.2f}원, 수량: {buy_volume} XRP")
        if response_buy and 'error' in response_buy:
            logger.error(f"오류 응답: {response_buy['error']}")

def execute_sell_order(trading_manager, buy_order_id):
    order_pair = trading_manager.get_order_pair(buy_order_id)
    if not order_pair:
        logger.error(f"매수 주문 {buy_order_id}에 대한 정보를 찾을 수 없습니다.")
        return

    buy_order = order_pair.buy_order
    buy_price = float(buy_order['price'])
    buy_volume = float(buy_order['volume'])

    orderbook = fetch_orderbook('KRW-XRP')
    if not orderbook:
        logger.error("오더북 조회 실패")
        return

    sell_price = calculate_optimal_sell_price(buy_price, orderbook)

    if is_profit_possible(buy_price, sell_price):
        response_sell = place_order('KRW-XRP', 'ask', buy_volume, sell_price)
        if response_sell and 'uuid' in response_sell:
            sell_order = {'uuid': response_sell['uuid'], 'price': sell_price, 'volume': buy_volume}
            trading_manager.update_sell_order(buy_order_id, sell_order)
            logger.info(f"매도 주문 생성됨: 주문 ID {sell_order['uuid']}, 가격: {sell_price:.2f}원, 수량: {buy_volume} XRP")
        else:
            logger.error(f"매도 주문 생성 실패: 가격 {sell_price:.2f}원, 수량: {buy_volume} XRP")
    else:
        logger.info(f"수익 불가능: 매수가 {buy_price:.2f}원, 현재 최적 매도가 {sell_price:.2f}원")

def monitor_orders(trading_manager):
    global available_seed
    
    while True:
        for buy_order_id in list(trading_manager.order_pairs.keys()):
            buy_status = check_order_status(buy_order_id)
            
            if buy_status is None:
                logger.error(f"주문 상태 확인 실패: {buy_order_id}")
                continue
            
            order_pair = trading_manager.get_order_pair(buy_order_id)
            if order_pair is None:
                logger.error(f"주문 쌍을 찾을 수 없음: {buy_order_id}")
                continue
            
            order_pair.buy_order['state'] = buy_status.get('state', 'unknown')

            if buy_status.get('state') == 'done':
                if not order_pair.sell_order:
                    execute_sell_order(trading_manager, buy_order_id)
                elif order_pair.sell_order:
                    sell_status = check_order_status(order_pair.sell_order['uuid'])
                    if sell_status:
                        order_pair.sell_order['state'] = sell_status.get('state', 'unknown')
                        if sell_status.get('state') == 'done':
                            buy_amount = float(buy_status['price']) * float(buy_status['volume'])
                            sell_amount = float(sell_status['price']) * float(sell_status['volume'])
                            profit = sell_amount - buy_amount
                            logger.info(f"거래 완료: 매수가 {float(buy_status['price']):.2f}원, 매도가 {float(sell_status['price']):.2f}원, 수익 {profit:.2f}원")
                            
                            with totals_lock:
                                available_seed += sell_amount
                            
                            trading_manager.remove_order_pair(buy_order_id)
            elif buy_status.get('state') in ['cancel', 'fail']:
                if order_pair.sell_order:
                    cancel_order(order_pair.sell_order['uuid'], trading_manager)
                
                trading_manager.remove_reserved_amount(buy_order_id)
                trading_manager.remove_order_pair(buy_order_id)

        time.sleep(1)

def adjust_buy_orders(trading_manager):
    orderbook = fetch_orderbook('KRW-XRP')
    if not orderbook:
        logger.error("오더북 조회 실패")
        return

    optimal_buy_price = calculate_optimal_buy_price(orderbook)

    for buy_order_id, order_pair in list(trading_manager.order_pairs.items()):
        if order_pair.sell_order is None:
            current_status = check_order_status(buy_order_id)
            if current_status is None or current_status.get('state') in ['done', 'cancel', 'fail']:
                logger.info(f"주문 {buy_order_id}의 상태가 {current_status.get('state', 'unknown')}입니다. 조정하지 않습니다.")
                continue

            current_buy_price = float(order_pair.buy_order['price'])
            if abs(optimal_buy_price - current_buy_price) > get_tick_size(current_buy_price):
                cancel_result = cancel_order(buy_order_id, trading_manager)
                if cancel_result:
                    logger.info(f"기존 매수 주문 취소: ID {buy_order_id}")
                    
                    new_buy_volume = float(order_pair.buy_order['volume'])
                    response_buy = place_order('KRW-XRP', 'bid', new_buy_volume, optimal_buy_price)
                    if response_buy and 'uuid' in response_buy:
                        new_buy_order_id = response_buy['uuid']
                        new_buy_order = {'uuid': new_buy_order_id, 'price': optimal_buy_price, 'volume': new_buy_volume}
                        
                        trading_manager.remove_order_pair(buy_order_id)
                        trading_manager.add_buy_order(new_buy_order)
                        
                        logger.info(f"새 매수 주문 생성: ID {new_buy_order_id}, 가격: {optimal_buy_price:.2f}원, 수량: {new_buy_volume} XRP")
                    else:
                        logger.error("Upbit API 새 매수 주문 실패")
                else:
                    logger.info(f"매수 주문 취소 불필요 또는 실패: ID {buy_order_id}")

def adjust_sell_orders(trading_manager):
    orderbook = fetch_orderbook('KRW-XRP')
    if not orderbook:
        logger.error("오더북 조회 실패")
        return

    for buy_order_id, order_pair in list(trading_manager.order_pairs.items()):
        if order_pair.sell_order:
            buy_price = float(order_pair.buy_order['price'])
            current_sell_price = float(order_pair.sell_order['price'])
            optimal_sell_price = calculate_optimal_sell_price(buy_price, orderbook)

            if abs(optimal_sell_price - current_sell_price) > get_tick_size(current_sell_price):
                cancel_result = cancel_order(order_pair.sell_order['uuid'], trading_manager)
                if cancel_result:
                    logger.info(f"기존 매도 주문 취소: ID {order_pair.sell_order['uuid']}")
                    
                    new_sell_volume = float(order_pair.sell_order['volume'])
                    response_sell = place_order('KRW-XRP', 'ask', new_sell_volume, optimal_sell_price)
                    if response_sell and 'uuid' in response_sell:
                        new_sell_order = {'uuid': response_sell['uuid'], 'price': optimal_sell_price, 'volume': new_sell_volume}
                        trading_manager.update_sell_order(buy_order_id, new_sell_order)
                        logger.info(f"새 매도 주문 생성: ID {new_sell_order['uuid']}, 가격: {optimal_sell_price:.2f}원, 수량: {new_sell_volume} XRP")
                    else:
                        logger.error("Upbit API 새 매도 주문 실패")
                else:
                    logger.error(f"매도 주문 취소 실패: ID {order_pair.sell_order['uuid']}")

def check_trading_status():
    global consecutive_failures, available_seed, new_buy_orders_allowed
    while True:
        try:
            current_balance = get_current_krw_balance()
            logger.info(f"현재 KRW 잔고: {current_balance:.2f}원")
            logger.info(f"현재 사용 가능한 시드 금액: {available_seed:.2f}원")
            
            if available_seed <= 0:
                logger.warning("시드 머니를 모두 사용했습니다. 신규 매수 주문을 중지합니다.")
                disable_new_buy_orders("시드 머니 소진")
            elif available_seed >= MIN_SEED_AMOUNT and not new_buy_orders_allowed:
                logger.info(f"사용 가능한 시드 금액이 {MIN_SEED_AMOUNT}원 이상입니다. 신규 매수 주문을 다시 활성화합니다.")
                enable_new_buy_orders()
            
            if consecutive_failures > 0:
                logger.warning(f"현재 연속 실패 횟수: {consecutive_failures}")
            
        except Exception as e:
            logger.error(f"거래 상태 체크 중 오류 발생: {e}", exc_info=True)
        
        time.sleep(60)  # 1분마다 체크

def pause_trading():
    global trading_paused
    with trading_pause_lock:
        trading_paused = True
    logger.info("거래가 일시 중지되었습니다.")

def resume_trading():
    global trading_paused
    with trading_pause_lock:
        trading_paused = False
    logger.info("거래가 재개되었습니다.")

def disable_new_buy_orders(reason=""):
    global new_buy_orders_allowed
    with new_buy_orders_lock:
        new_buy_orders_allowed = False
    logger.warning(f"신규 매수 주문이 비활성화되었습니다. 이유: {reason}")

def enable_new_buy_orders():
    global new_buy_orders_allowed
    with new_buy_orders_lock:
        new_buy_orders_allowed = True
    logger.info("신규 매수 주문이 다시 활성화되었습니다.")

def get_current_price(market):
    url = f"{server_url}/v1/ticker"
    params = {'markets': market}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return float(response.json()[0]['trade_price'])
    except Exception as e:
        logger.error(f"현재 가격 조회 중 오류 발생: {e}")
        return None

def check_stop_loss(trading_manager, buy_order_id):
    order_pair = trading_manager.get_order_pair(buy_order_id)
    if not order_pair:
        return

    buy_price = float(order_pair.buy_order['price'])
    current_price = get_current_price('KRW-XRP')
    
    if current_price is None:
        logger.error("현재 가격 조회 실패로 손절 체크를 건너뜁니다.")
        return

    loss_percentage = (current_price - buy_price) / buy_price

    if loss_percentage < -0.03:  # 3% 손실 시 손절
        logger.warning(f"손절 조건 충족: 매수가 {buy_price:.2f}, 현재가 {current_price:.2f}, 손실률 {loss_percentage:.2%}")
        execute_market_sell(trading_manager, buy_order_id)

def execute_market_sell(trading_manager, buy_order_id):
    order_pair = trading_manager.get_order_pair(buy_order_id)
    if not order_pair:
        return

    volume = float(order_pair.buy_order['volume'])
    response_sell = place_order('KRW-XRP', 'ask', volume, 0, ord_type='market')
    
    if response_sell and 'uuid' in response_sell:
        logger.info(f"시장가 매도 주문 실행: ID {response_sell['uuid']}, 수량: {volume} XRP")
        
        sell_status = check_order_status(response_sell['uuid'])
        if sell_status and sell_status.get('state') == 'done':
            sell_amount = float(sell_status['price']) * float(sell_status['volume'])
            logger.info(f"시장가 매도 완료: 매도가 {float(sell_status['price']):.2f}원, 총액 {sell_amount:.2f}원")
            
            with totals_lock:
                global available_seed
                available_seed += sell_amount
            
            trading_manager.remove_order_pair(buy_order_id)
        else:
            logger.error("시장가 매도 주문 실패 또는 미체결")
    else:
        logger.error("시장가 매도 주문 생성 실패")

def real_time_order_scheduler(trading_manager):
    global consecutive_failures
    while True:
        try:
            with trading_pause_lock:
                if not trading_paused:
                    execute_buy_order(trading_manager)
                    adjust_buy_orders(trading_manager)
                    adjust_sell_orders(trading_manager)
            consecutive_failures = 0
        except Exception as e:
            logger.error(f"실시간 주문 스케줄링 중 오류 발생: {e}", exc_info=True)
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                logger.critical(f"연속 {max_consecutive_failures}회 실패. 거래를 일시 중지합니다.")
                pause_trading()
        time.sleep(0.2)

def main():
    trading_manager = TradingManager()
    logger.info("트레이딩 봇 시작")

    # 모니터링 및 주문 처리를 위한 스레드 시작
    threading.Thread(target=monitor_orders, args=(trading_manager,), daemon=True).start()
    threading.Thread(target=real_time_order_scheduler, args=(trading_manager,), daemon=True).start()
    threading.Thread(target=check_trading_status, daemon=True).start()

    try:
        # 메인 스레드를 계속 실행 상태로 유지
        while True:
            time.sleep(60)  # 1분마다 깨어나서 계속 실행 중임을 로그에 기록
            logger.info("트레이딩 봇 실행 중...")
    except KeyboardInterrupt:
        logger.info("트레이딩 봇 종료 요청을 받았습니다. 종료합니다.")
    except Exception as e:
        logger.error(f"예기치 않은 오류 발생: {e}", exc_info=True)
    finally:
        logger.info("트레이딩 봇을 종료합니다.")

if __name__ == "__main__":
    main()