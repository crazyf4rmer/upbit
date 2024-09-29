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

# .env 파일 로드 (스크립트 파일의 동일 디렉토리에 위치)
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

# Upbit API Keys 설정 (환경 변수에서 불러오기)
access_key = os.getenv('UPBIT_OPEN_API_ACCESS_KEY')
secret_key = os.getenv('UPBIT_OPEN_API_SECRET_KEY')
server_url = os.getenv('UPBIT_OPEN_API_SERVER_URL', 'https://api.upbit.com')

max_consecutive_failures = 3
consecutive_failures = 0

# 시드 금액 설정
SEED_AMOUNT = 50000.0  # 단위: KRW

# 로깅 설정
logger = logging.getLogger('trading_bot')
logger.setLevel(logging.DEBUG)

# 파일 핸들러 (DEBUG 이상) - 로그 파일 회전 설정 추가
file_handler = RotatingFileHandler('trading_bot.log', maxBytes=10*1024*1024, backupCount=5)
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

# 전역 변수 및 데이터 구조
trade_in_progress = False
trade_lock = threading.Lock()
trading_paused = False
trading_pause_lock = threading.Lock()
total_invested = 0.0
available_seed = SEED_AMOUNT
totals_lock = threading.Lock()
file_lock = threading.Lock()

class OrderPair:
    def __init__(self, buy_order, initial_sell_price):
        self.buy_order = buy_order
        self.sell_order = None
        self.initial_sell_price = initial_sell_price
        self.current_sell_price = initial_sell_price

class TradingManager:
    def __init__(self):
        self.order_pairs = {}
        self.lock = threading.Lock()

    def add_buy_order(self, buy_order, initial_sell_price):
        with self.lock:
            self.order_pairs[buy_order['order_id']] = OrderPair(buy_order, initial_sell_price)

    def update_sell_order(self, buy_order_id, sell_order):
        with self.lock:
            if buy_order_id in self.order_pairs:
                self.order_pairs[buy_order_id].sell_order = sell_order

    def get_order_pair(self, buy_order_id):
        with self.lock:
            return self.order_pairs.get(buy_order_id)

    def remove_order_pair(self, buy_order_id):
        with self.lock:
            if buy_order_id in self.order_pairs:
                del self.order_pairs[buy_order_id]

@contextmanager
def open_order_lock():
    file_lock.acquire()
    try:
        yield
    finally:
        file_lock.release()

def get_tick_size(price):
    return 0.1

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
        'nonce': str(uuid.uuid4()),
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
    except Exception as e:
        logger.error(f"주문 실행 중 오류 발생: {e}")
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

def cancel_order(order_id):
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
        response = requests.delete(f"{server_url}/v1/order", params=query, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"주문 취소 중 오류 발생: {e}")
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

def calculate_optimal_sell_price(buy_price, orderbook, min_profit_rate=0.002):
    min_sell_price = buy_price * (1 + min_profit_rate)
    asks = orderbook['orderbook_units']
    valid_asks = [ask for ask in asks if float(ask['ask_price']) >= min_sell_price]
    
    if not valid_asks:
        return min_sell_price

    volume_at_price = defaultdict(float)
    for ask in valid_asks:
        volume_at_price[float(ask['ask_price'])] += float(ask['ask_size'])
    
    optimal_price = max(volume_at_price, key=volume_at_price.get)
    return optimal_price

def execute_buy_order(trading_manager):
    global available_seed, total_invested
    
    orderbook = fetch_orderbook('KRW-XRP')
    if not orderbook:
        logger.error("오더북 조회 실패")
        return

    bids = orderbook['orderbook_units']
    best_bid = float(bids[0]['bid_price'])
    buy_price = best_bid + get_tick_size(best_bid)
    buy_volume = 10  # 예시 수량, 실제로는 적절한 수량 계산 로직이 필요합니다.

    if available_seed >= buy_price * buy_volume:
        response_buy = place_order('KRW-XRP', 'bid', buy_volume, buy_price)
        if response_buy and 'uuid' in response_buy:
            buy_order_id = response_buy['uuid']
            buy_order = {'order_id': buy_order_id, 'price': buy_price, 'volume': buy_volume}
            initial_sell_price = calculate_optimal_sell_price(buy_price, orderbook)
            trading_manager.add_buy_order(buy_order, initial_sell_price)
            
            with totals_lock:
                total_invested += buy_price * buy_volume
                available_seed -= buy_price * buy_volume
            
            logger.info(f"매수 주문 생성됨: 주문 ID {buy_order_id}, 가격: {buy_price:.2f}원, 수량: {buy_volume} XRP")
        else:
            logger.debug(f"매수 주문 생성 실패: 가격 {buy_price:.2f}원, 수량: {buy_volume} XRP")
    else:
        logger.debug(f"시드 금액 부족: 필요 금액 {buy_price * buy_volume:.2f}원, 사용 가능 금액 {available_seed:.2f}원")

def adjust_sell_orders(trading_manager):
    orderbook = fetch_orderbook('KRW-XRP')
    if not orderbook:
        logger.error("오더북 조회 실패")
        return

    for buy_order_id, order_pair in list(trading_manager.order_pairs.items()):
        buy_order = order_pair.buy_order
        current_sell_price = order_pair.current_sell_price
        optimal_sell_price = calculate_optimal_sell_price(buy_order['price'], orderbook)

        if abs(optimal_sell_price - current_sell_price) > get_tick_size(current_sell_price):
            if order_pair.sell_order:
                cancel_order(order_pair.sell_order['order_id'])
                logger.info(f"기존 매도 주문 취소: ID {order_pair.sell_order['order_id']}")

            response_sell = place_order('KRW-XRP', 'ask', buy_order['volume'], optimal_sell_price)
            if response_sell and 'uuid' in response_sell:
                new_sell_order = {'order_id': response_sell['uuid'], 'price': optimal_sell_price, 'volume': buy_order['volume']}
                trading_manager.update_sell_order(buy_order_id, new_sell_order)
                order_pair.current_sell_price = optimal_sell_price
                logger.info(f"새 매도 주문 생성: ID {new_sell_order['order_id']}, 가격: {optimal_sell_price:.2f}원, 수량: {buy_order['volume']} XRP")
            else:
                logger.error("Upbit API 매도 주문 실패")

def monitor_orders(trading_manager):
    global total_invested, available_seed
    
    while True:
        for buy_order_id, order_pair in list(trading_manager.order_pairs.items()):
            buy_status = check_order_status(order_pair.buy_order['order_id'])
            if buy_status['state'] == 'done':
                if not order_pair.sell_order:
                    # 매수 주문이 체결되었지만 아직 매도 주문이 없는 경우
                    execute_sell_order(trading_manager, buy_order_id)
                else:
                    sell_status = check_order_status(order_pair.sell_order['order_id'])
                    if sell_status['state'] == 'done':
                        # 매수-매도 주문 쌍이 모두 체결된 경우
                        profit = (order_pair.sell_order['price'] - order_pair.buy_order['price']) * order_pair.buy_order['volume']
                        logger.info(f"거래 완료: 매수가 {order_pair.buy_order['price']:.2f}원, 매도가 {order_pair.sell_order['price']:.2f}원, 수익 {profit:.2f}원")
                        
                        with totals_lock:
                            total_invested -= order_pair.buy_order['price'] * order_pair.buy_order['volume']
                            available_seed += order_pair.sell_order['price'] * order_pair.sell_order['volume']
                        
                        trading_manager.remove_order_pair(buy_order_id)
            elif buy_status['state'] in ['cancel', 'fail']:
                # 매수 주문이 취소되거나 실패한 경우
                if order_pair.sell_order:
                    cancel_order(order_pair.sell_order['order_id'])
                
                with totals_lock:
                    total_invested -= order_pair.buy_order['price'] * order_pair.buy_order['volume']
                    available_seed += order_pair.buy_order['price'] * order_pair.buy_order['volume']
                
                trading_manager.remove_order_pair(buy_order_id)

        time.sleep(1)  # 1초마다 주문 상태 확인

def execute_sell_order(trading_manager, buy_order_id):
    order_pair = trading_manager.get_order_pair(buy_order_id)
    if not order_pair:
        return

    orderbook = fetch_orderbook('KRW-XRP')
    if not orderbook:
        logger.error("오더북 조회 실패")
        return

    optimal_sell_price = calculate_optimal_sell_price(order_pair.buy_order['price'], orderbook)
    response_sell = place_order('KRW-XRP', 'ask', order_pair.buy_order['volume'], optimal_sell_price)
    
    if response_sell and 'uuid' in response_sell:
        sell_order = {'order_id': response_sell['uuid'], 'price': optimal_sell_price, 'volume': order_pair.buy_order['volume']}
        trading_manager.update_sell_order(buy_order_id, sell_order)
        logger.info(f"매도 주문 생성: ID {sell_order['order_id']}, 가격: {optimal_sell_price:.2f}원, 수량: {order_pair.buy_order['volume']} XRP")
    else:
        logger.error("Upbit API 매도 주문 실패")

def check_trading_status():
    global consecutive_failures
    while True:
        try:
            current_balance = get_current_krw_balance()
            logger.info(f"현재 KRW 잔고: {current_balance:.2f}원")
            logger.info(f"현재 총 투자 금액: {total_invested:.2f}원")
            logger.info(f"남은 시드 금액: {available_seed:.2f}원")
            
            if available_seed <= 0:
                logger.warning("시드 머니를 모두 사용했습니다. 거래를 일시 중지합니다.")
                pause_trading()
            
            # 연속 실패 횟수 확인
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

def real_time_order_scheduler(trading_manager):
    global consecutive_failures
    while True:
        try:
            with trading_pause_lock:
                if not trading_paused:
                    execute_buy_order(trading_manager)  # 매수 주문 실행
                    adjust_sell_orders(trading_manager)   # 매도 주문 조정
            consecutive_failures = 0  # 성공적으로 실행되면 연속 실패 카운트 리셋
        except Exception as e:
            logger.error(f"실시간 주문 스케줄링 중 오류 발생: {e}", exc_info=True)
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                logger.critical(f"연속 {max_consecutive_failures}회 실패. 거래를 일시 중지합니다.")
                pause_trading()
        time.sleep(0.2)  # 0.2초 간격으로 체크 (5번/초)

def main():
    trading_manager = TradingManager()
    logger.info("트레이딩 봇 시작")

    threading.Thread(target=monitor_orders, args=(trading_manager,), daemon=True).start()
    threading.Thread(target=real_time_order_scheduler, args=(trading_manager,), daemon=True).start()
    threading.Thread(target=check_trading_status, daemon=True).start()

    while True:
        command = input("명령을 입력하세요 (pause/resume/exit): ").strip().lower()
        if command == 'pause':
            pause_trading()
        elif command == 'resume':
            resume_trading()
        elif command == 'exit':
            logger.info("트레이딩 봇을 종료합니다.")
            break
        else:
            logger.info("알 수 없는 명령입니다.")

if __name__ == "__main__":
    main()