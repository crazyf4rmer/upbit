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

# 파일 핸들러 (DEBUG 이상) - 로그 파일 회전 설정 추가
from logging.handlers import RotatingFileHandler

file_handler = RotatingFileHandler('trading_bot.log', maxBytes=10*1024*1024, backupCount=5)  # 10MB 단위로 최대 5개 백업
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 콘솔 핸들러 (INFO 이상) - 색상 제거
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# 시드 금액 설정
SEED_AMOUNT = 50000.0  # 단위: KRW

# 틱 사이즈 고정 (리플은 약 800원 대이므로 0.1으로 고정)
def get_tick_size(price):
    """
    리플의 틱 사이즈를 고정합니다.
    """
    return 0.1

# 수익성 검증 함수
def is_profit_possible(buy_price, sell_price, fee_rate=0.0005):
    """
    실제 수익이 0보다 큰지 확인합니다.
    실제 수익 = 매도가 - 매수가 - (매도가 + 매수가) * 0.0005
    """
    profit = sell_price - buy_price - (sell_price + buy_price) * fee_rate
    logger.debug(f"Buy Price: {buy_price}, Sell Price: {sell_price}, Profit: {profit}")
    return profit > 0

# 전역 변수 및 데이터 구조
sell_orders = {}  # {buy_order_id: {'sell_order_id': sell_id, 'buy_price': buy_price, 'volume': volume, 'sell_price': sell_price}}
buy_orders = {}   # {buy_order_id: {'price': buy_price, 'volume': volume, 'timestamp': datetime_object, 'amount': amount}}
last_order_time = datetime.min.replace(tzinfo=timezone.utc)  # 마지막 주문 시간 초기화
trade_session_counter = 1  # 거래 세션 번호 초기화
trade_in_progress = False  # 현재 거래 진행 중 여부
trade_lock = threading.Lock()  # 거래 동기화를 위한 락

# 거래 일시 중지 플래그
trading_paused = False
trading_pause_lock = threading.Lock()

# 누적 변수 및 락 추가
total_buys = 0.0  # 총 매수 금액
total_sells = 0.0  # 총 매도 금액
cumulative_profit = 0.0  # 누적 순이익
total_invested = 0.0  # 누적 투자 금액
available_seed = SEED_AMOUNT  # 사용 가능한 시드 금액
totals_lock = threading.Lock()  # 누적 변수 동기화를 위한 락

# 매도 주문 가격 조정을 위한 락 추가
adjust_sell_lock = threading.Lock()

# 파일 핸들러 (잠금 메커니즘)
file_lock = threading.Lock()

@contextmanager
def open_order_lock():
    file_lock.acquire()
    try:
        yield
    finally:
        file_lock.release()

# 시간 파싱 함수 추가
def parse_created_at(created_at_str):
    """
    Upbit API의 'created_at' 필드를 파싱하여 timezone-aware datetime 객체로 변환합니다.
    """
    try:
        if created_at_str.endswith('Z'):
            created_at_str = created_at_str.replace('Z', '+00:00')
        return datetime.fromisoformat(created_at_str)
    except ValueError as e:
        logger.debug(f"created_at 파싱 중 오류 발생: {e} - 입력값: {created_at_str}")
        return None

# 현재 KRW 잔액을 조회하는 함수 추가
def get_current_krw_balance():
    """
    Upbit API의 /v1/accounts 엔드포인트를 활용하여 현재 KRW 잔액을 조회합니다.
    """
    url = f"{server_url}/v1/accounts"
    query = {}
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
        logger.debug(f"잔고 조회 JWT 인코딩 오류: {e}")
        return 0.0

    authorization = f'Bearer {jwt_token}'
    headers = {
        "Authorization": authorization
    }

    try:
        response = requests.get(url, params=query, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        for account in data:
            if account.get('currency') == 'KRW':
                balance = float(account.get('balance', 0))
                logger.debug(f"현재 KRW 잔고: {balance}원")
                return balance
        return 0.0
    except requests.exceptions.Timeout:
        logger.debug("잔고 조회 요청 타임아웃")
    except Exception as e:
        logger.debug(f"잔고 조회 중 오류 발생: {e}")
        return 0.0

# 매수/매도 주문 실행 함수
def place_order(market, side, volume, price, linked_order_id=None, ord_type='limit'):
    global available_seed, total_invested

    params = {
        'market': market,
        'side': side,
        'volume': str(volume),
        'price': str(price),
        'ord_type': ord_type,
    }

    if linked_order_id:
        params['linked_order_id'] = linked_order_id

    query_string = unquote(urlencode(params, doseq=True)).encode("utf-8")

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
        logger.debug(f"JWT 인코딩 오류: {e}")
        return None

    authorization = f'Bearer {jwt_token}'
    headers = {
        "Authorization": authorization
    }

    try:
        response = requests.post(f"{server_url}/v1/orders", json=params, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        order_id = data.get('uuid')
        if order_id:
            if side == 'bid':
                order_amount = float(price) * float(volume)
                with totals_lock:
                    if available_seed >= order_amount:
                        available_seed -= order_amount
                        total_invested += order_amount
                        buy_orders[order_id] = {
                            'price': float(price),
                            'volume': float(volume),
                            'timestamp': datetime.now(timezone.utc),
                            'amount': order_amount
                        }
                        logger.debug(f"매수 주문 생성됨: {price:.2f}원, 수량: {volume} XRP, 주문 ID: {order_id}, 투자 금액: {order_amount:.2f}원")
                        logger.debug(f"남은 시드 금액: {available_seed:.2f}원, 총 투자 금액: {total_invested:.2f}원")
                    else:
                        logger.debug(f"시드 금액 부족으로 매수 주문 실패: 필요 금액 {order_amount:.2f}원, 가용 금액 {available_seed:.2f}원")
                        return None
            elif side == 'ask' and linked_order_id:
                sell_orders[linked_order_id] = {
                    'sell_order_id': order_id,
                    'buy_price': float(price),
                    'volume': float(volume),
                    'sell_price': float(price)
                }
                logger.debug(f"매도 주문 생성됨: {float(price):.2f}원, 수량: {volume} XRP, 매수 주문 ID: {linked_order_id}, 매도 주문 ID: {order_id}")
            return data
        return data
    except requests.exceptions.Timeout:
        logger.debug(f"주문 요청 타임아웃: {side} 주문 - 가격: {price}, 수량: {volume}")
    except requests.exceptions.HTTPError as http_err:
        logger.debug(f"HTTP 오류 발생: {http_err} - 응답: {response.text}")
    except Exception as e:
        logger.debug(f"주문 실행 중 오류 발생: {e}")
        logger.debug(f"쿼리 문자열: {query_string}")
        logger.debug(f"쿼리 해시: {query_hash}")

    return None

# 매도 주문 별도의 함수
def place_sell_order_on_buy(order_id, buy_price, volume, sell_price):
    global trade_in_progress

    if not is_profit_possible(buy_price, sell_price):
        logger.debug(f"실제 수익이 0보다 작아 매도 주문을 실행하지 않습니다: 매수 가격 {buy_price}원, 매도 가격 {sell_price}원")
        with trade_lock:
            trade_in_progress = False
        return None

    tick_size = get_tick_size(sell_price)
    rounded_sell_price = math.ceil(sell_price / tick_size) * tick_size

    logger.debug(f"매도 가격 계산: {sell_price:.2f}원")
    logger.debug(f"반올림된 매도 가격: {rounded_sell_price:.2f}원 (틱 사이즈: {tick_size})")

    response_sell = place_order('KRW-XRP', 'ask', volume, rounded_sell_price, linked_order_id=order_id)

    if response_sell and 'uuid' in response_sell:
        sell_order_id = response_sell['uuid']
        logger.debug(f"매도 주문 실행됨: {rounded_sell_price:.2f}원, 수량: {volume} XRP (매수 주문 ID: {order_id}, 매도 주문 ID: {sell_order_id})")
    else:
        logger.debug(f"매도 주문 실행 실패: {rounded_sell_price:.2f}원, 수량: {volume} XRP (매수 주문 ID: {order_id})")
        with trade_lock:
            trade_in_progress = False

    return response_sell

# 주문 상태 확인 함수
def check_order_status(order_id, side):
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
        logger.debug(f"주문 상태 확인 JWT 인코딩 오류: {e}")
        return None

    authorization = f'Bearer {jwt_token}'
    headers = {
        "Authorization": authorization
    }

    try:
        response = requests.get(url, params=query, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.Timeout:
        logger.debug(f"주문 상태 확인 요청 타임아웃: 주문 ID {order_id}")
    except requests.exceptions.HTTPError as http_err:
        logger.debug(f"주문 상태 확인 중 HTTP 오류 발생: {http_err} - 응답: {response.text}")
    except Exception as e:
        logger.debug(f"주문 상태 확인 중 오류 발생: {e}")

    return None

# 주문 취소 함수
def cancel_order(order_id):
    global available_seed, total_invested

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
        logger.debug(f"주문 취소 JWT 인코딩 오류: {e}")
        return None

    authorization = f'Bearer {jwt_token}'
    headers = {
        "Authorization": authorization
    }

    try:
        response = requests.delete(url, params=query, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        logger.info(f"주문 취소됨: 주문 ID {order_id}")

        if order_id in buy_orders:
            buy_info = buy_orders[order_id]
            buy_amount = buy_info['amount']

            with totals_lock:
                available_seed += buy_amount
                total_invested -= buy_amount
                logger.debug(f"매수 주문 취소로 인한 시드 금액 복구: {buy_amount:.2f}원")
                logger.debug(f"남은 시드 금액: {available_seed:.2f}원, 총 투자 금액: {total_invested:.2f}원")

            del buy_orders[order_id]

        return data
    except requests.exceptions.Timeout:
        logger.debug(f"주문 취소 요청 타임아웃: 주문 ID {order_id}")
    except requests.exceptions.HTTPError as http_err:
        logger.debug(f"주문 취소 중 HTTP 오류 발생: {http_err} - 응답: {response.text}")
    except Exception as e:
        logger.debug(f"주문 취소 중 오류 발생: {e}")
        logger.debug(f"쿼리 문자열: {query_string}")
        logger.debug(f"쿼리 해시: {query_hash}")

    return None

# 모든 미체결 주문 가져오기 함수
def get_open_orders():
    url = f"{server_url}/v1/orders"
    query = {
        'state': 'wait'
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
        logger.debug(f"모든 미체결 주문 가져오기 JWT 인코딩 오류: {e}")
        return []

    authorization = f'Bearer {jwt_token}'
    headers = {
        "Authorization": authorization
    }

    try:
        response = requests.get(url, params=query, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.Timeout:
        logger.debug("모든 미체결 주문 가져오기 요청 타임아웃")
    except Exception as e:
        logger.debug(f"모든 미체결 주문 가져오기 중 오류 발생: {e}")
        return []

# 주문서 데이터 처리 함수
def process_orderbook(orderbook):
    try:
        orderbook_units = orderbook['orderbook_units']
        bids = []
        asks = []
        for unit in orderbook_units:
            bids.append({'price': float(unit['bid_price']), 'size': float(unit['bid_size'])})
            asks.append({'price': float(unit['ask_price']), 'size': float(unit['ask_size'])})
        bids.sort(key=lambda x: x['price'], reverse=True)
        asks.sort(key=lambda x: x['price'])
        return bids, asks
    except (KeyError, TypeError, IndexError, ValueError) as e:
        logger.debug(f"주문서 데이터 처리 중 오류 발생: {e}")
        return None, None

# 실시간 시장 데이터 가져오기 함수
def fetch_orderbook(market):
    url = f"{server_url}/v1/orderbook"
    params = {'markets': market}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        orderbook_data = response.json()
        return orderbook_data[0] if orderbook_data else None
    except requests.exceptions.Timeout:
        logger.debug(f"주문서 가져오기 요청 타임아웃: 시장 {market}")
    except Exception as e:
        logger.debug(f"주문서 가져오기 중 오류 발생: {e}")
        return None

# 현재 시장 가격 가져오기 함수 추가
def get_current_market_price(market):
    orderbook = fetch_orderbook(market)
    if not orderbook:
        logger.debug("현재 시장 가격 가져오기 실패.")
        return None
    bids, asks = process_orderbook(orderbook)
    if not bids or not asks:
        logger.debug("현재 시장 호가 처리 실패.")
        return None
    best_bid = bids[0]['price']
    best_ask = asks[0]['price']
    current_price = (best_bid + best_ask) / 2
    return current_price

# 주문서 데이터 취소 후 매도 주문 생성 함수
def wait_and_place_sell_order(buy_order_id, buy_price, volume):
    global trade_in_progress

    while True:
        status = check_order_status(buy_order_id, 'bid')
        if status:
            state = status.get('state')
            if state == 'done':
                logger.debug(f"매수 주문 체결 확인 후 0.2초 대기: 주문 ID {buy_order_id}")
                time.sleep(0.2)

                orderbook = fetch_orderbook('KRW-XRP')
                if not orderbook:
                    logger.debug("오더북 가져오기 실패. 매도 주문을 생성하지 않습니다.")
                    with trade_lock:
                        trade_in_progress = False
                    break

                bids, asks = process_orderbook(orderbook)
                if not bids or not asks:
                    logger.debug("오더북 호가 처리 실패. 매도 주문을 생성하지 않습니다.")
                    with trade_lock:
                        trade_in_progress = False
                    break

                highest_ask = max(asks, key=lambda x: x['size'])
                sell_price = highest_ask['price'] - get_tick_size(highest_ask['price'])

                if len(asks) >= 2:
                    gap = asks[1]['price'] - asks[0]['price']
                    if gap < get_tick_size(highest_ask['price']):
                        logger.debug("호가 갭이 최소 틱보다 작아 매도 주문을 생성하지 않습니다.")
                        with trade_lock:
                            trade_in_progress = False
                        break
                else:
                    logger.debug("호가 갭 계산을 위한 충분한 매도 호가가 없습니다.")
                    with trade_lock:
                        trade_in_progress = False
                    break

                current_best_ask_before = highest_ask['price']
                time.sleep(0.2)
                orderbook_after = fetch_orderbook('KRW-XRP')
                if not orderbook_after:
                    logger.debug("오더북 가져오기 실패. 매도 주문을 생성하지 않습니다.")
                    with trade_lock:
                        trade_in_progress = False
                    break

                bids_after, asks_after = process_orderbook(orderbook_after)
                if not bids_after or not asks_after:
                    logger.debug("오더북 호가 처리 실패. 매도 주문을 생성하지 않습니다.")
                    with trade_lock:
                        trade_in_progress = False
                    break

                highest_ask_after = max(asks_after, key=lambda x: x['size'])
                if highest_ask_after['price'] != current_best_ask_before:
                    logger.debug("매도 가격이 변동되어 매도 주문을 생성하지 않습니다.")
                    with trade_lock:
                        trade_in_progress = False
                    break

                response_sell = place_sell_order_on_buy(buy_order_id, buy_price, volume, sell_price)
                if response_sell:
                    logger.info(f"매도 주문이 성공적으로 생성되었습니다: 주문 ID {response_sell.get('uuid')}")
                else:
                    logger.debug(f"매도 주문 생성 실패: 매수 주문 ID {buy_order_id}")
                break
            elif state in ['cancelled', 'failed']:
                logger.info(f"매수 주문이 취소되거나 실패했습니다: 주문 ID {buy_order_id}, 상태: {state}")
                with trade_lock:
                    trade_in_progress = False
                break
        time.sleep(0.2)

# 매수 주문 모니터링 및 조정 함수
def monitor_buy_orders():
    global buy_orders, trade_in_progress, available_seed, total_invested
    while True:
        try:
            time.sleep(0.2)
            with open_order_lock():
                for buy_id, buy_info in list(buy_orders.items()):
                    price = buy_info['price']
                    volume = buy_info['volume']
                    timestamp = buy_info['timestamp']
                    buy_amount = buy_info['amount']
                    
                    status = check_order_status(buy_id, 'bid')
                    if status:
                        state = status.get('state')
                        if state == 'done':
                            logger.info(f"매수 주문 체결됨: 주문 ID {buy_id}, 가격: {price:.2f}원, 수량: {volume} XRP")
                            threading.Thread(target=wait_and_place_sell_order, args=(buy_id, price, volume), daemon=True).start()
                            del buy_orders[buy_id]
                        elif state in ['cancelled', 'failed']:
                            logger.info(f"매수 주문 취소됨: 주문 ID {buy_id}, 상태: {state}")
                            
                            with totals_lock:
                                available_seed += buy_amount
                                total_invested -= buy_amount
                                logger.debug(f"매수 주문 취소로 인한 시드 금액 복구: {buy_amount:.2f}원")
                                logger.debug(f"남은 시드 금액: {available_seed:.2f}원, 총 투자 금액: {total_invested:.2f}원")
                            
                            del buy_orders[buy_id]
                            
                            with trade_lock:
                                trade_in_progress = False
                        elif state == 'wait':
                            orderbook = fetch_orderbook('KRW-XRP')
                            if not orderbook:
                                continue
                            bids, asks = process_orderbook(orderbook)
                            if not bids or not asks:
                                continue
                            
                            new_highest_bid = max(bids, key=lambda x: x['size'])
                            new_buy_price = new_highest_bid['price'] + get_tick_size(new_highest_bid['price'])
                            
                            if new_buy_price > price:
                                logger.info(f"매수 주문 가격 조정 필요: 기존 가격 {price:.2f}원 -> 새로운 가격 {new_buy_price:.2f}원")
                                cancel_order(buy_id)
                                with trade_lock:
                                    trade_in_progress = False
                                
                                time.sleep(0.2)
                                orderbook_after = fetch_orderbook('KRW-XRP')
                                if not orderbook_after:
                                    continue
                                bids_after, asks_after = process_orderbook(orderbook_after)
                                if not bids_after or not asks_after:
                                    continue
                                
                                current_best_bid_after = max(bids_after, key=lambda x: x['size'])['price']
                                if current_best_bid_after != new_highest_bid['price']:
                                    logger.debug("매수 가격이 변동되어 새로운 매수 주문을 생성하지 않습니다.")
                                    continue
                                
                                buy_amount = 10000.0  # 예: 10,000원
                                current_balance = get_current_krw_balance()

                                with totals_lock:
                                    remaining_seed = available_seed

                                if remaining_seed <= 0:
                                    logger.info(f"시드 금액 {SEED_AMOUNT}원을 모두 투자했습니다. 추가 매수는 더 이상 진행되지 않습니다.")
                                    with trade_lock:
                                        trade_in_progress = False
                                    continue

                                buy_amount = min(buy_amount, remaining_seed, current_balance)

                                if buy_amount < get_tick_size(new_buy_price):
                                    logger.debug(f"매수 금액이 최소 주문 금액보다 작아 매수 주문을 걸지 않습니다: {buy_amount:.2f}원")
                                    with trade_lock:
                                        trade_in_progress = False
                                    continue

                                volume = buy_amount / new_buy_price
                                volume = round(volume, 6)  # 소수점 자리수 조정

                                response_new_buy = place_order('KRW-XRP', 'bid', volume, new_buy_price)
                                if response_new_buy and 'uuid' in response_new_buy:
                                    new_buy_id = response_new_buy['uuid']
                                    buy_orders[new_buy_id] = {
                                        'price': new_buy_price,
                                        'volume': volume,
                                        'timestamp': datetime.now(timezone.utc),
                                        'amount': buy_amount
                                    }
                                    logger.info(f"새로운 매수 주문 생성됨: 주문 ID {new_buy_id}, 가격: {new_buy_price:.2f}원, 수량: {volume} XRP")
                                else:
                                    logger.debug(f"새로운 매수 주문 생성 실패: 가격 {new_buy_price:.2f}원, 수량: {volume} XRP")
        except Exception as e:
            logger.debug(f"매수 주문 모니터링 중 오류 발생: {e}")

# 매도 주문 상태 확인 및 관리 함수
def monitor_sell_orders():
    global sell_orders, trade_session_counter, trade_in_progress, trading_paused, total_buys, total_sells, cumulative_profit, total_invested, available_seed

    while True:
        try:
            time.sleep(0.2)  # 0.2초마다 매도 주문 상태 확인
            with open_order_lock():
                for buy_order_id, sell_info in list(sell_orders.items()):
                    sell_order_id = sell_info['sell_order_id']
                    buy_price = sell_info['buy_price']
                    volume = sell_info['volume']
                    sell_price = sell_info['sell_price']

                    # 주문 상태 확인
                    status = check_order_status(sell_order_id, 'ask')
                    if status:
                        state = status.get('state')
                        if state == 'done':
                            logger.info(f"매도 주문 체결됨: 주문 ID {sell_order_id}, 가격: {sell_price:.2f}원, 수량: {volume} XRP")

                            # 이익 계산 및 누적
                            buy_amount = buy_price * volume
                            sell_amount = sell_price * volume
                            profit = sell_amount - buy_amount - (buy_amount + sell_amount) * 0.0005 * 2  # 수수료 고려

                            with totals_lock:
                                total_buys += buy_amount
                                total_sells += sell_amount
                                cumulative_profit += profit
                                total_invested -= buy_amount
                                available_seed += sell_amount  # 매도 금액을 시드 금액에 추가

                            logger.info(f"거래 세션 {trade_session_counter}: 매수 {buy_amount:.2f}원, 매도 {sell_amount:.2f}원, 순이익 {profit:.2f}원")
                            logger.info(f"누적 순이익: {cumulative_profit:.2f}원")
                            logger.debug(f"남은 시드 금액: {available_seed:.2f}원, 총 투자 금액: {total_invested:.2f}원")

                            trade_session_counter += 1

                            # 매도 주문 완료 후 데이터 제거
                            del sell_orders[buy_order_id]
                            with trade_lock:
                                trade_in_progress = False
                        elif state in ['cancelled', 'failed']:
                            logger.info(f"매도 주문 취소됨 또는 실패: 주문 ID {sell_order_id}, 상태: {state}")

                            # 매도 주문이 취소되었으므로, 해당 매수분을 시장가로 매도할지 결정
                            # 여기서는 단순히 데이터만 삭제하고 다음 기회로 넘김
                            del sell_orders[buy_order_id]
                            with trade_lock:
                                trade_in_progress = False
        except Exception as e:
            logger.debug(f"매도 주문 모니터링 중 오류 발생: {e}")

# 매수 및 매도 주문 실행 함수 (오더북 기반 단타 매매)
def execute_orderbook_based_trading():
    global last_order_time, trade_in_progress, trading_paused, total_invested, available_seed
    market = 'KRW-XRP'  # 리플 시장 심볼로 변경

    current_time = datetime.now(timezone.utc)
    time_since_last_order = (current_time - last_order_time).total_seconds()

    if time_since_last_order < 0.2:  # 최소 0.2초 대기 (5번/초)
        logger.debug(f"주문 간 최소 0.2초 대기 중. 마지막 주문 이후 {time_since_last_order:.2f}초 경과.")
        return

    # 거래 진행 중인지 확인
    with trade_lock:
        if trade_in_progress:
            logger.debug("현재 거래가 진행 중입니다. 새 거래를 시작하지 않습니다.")
            return
        trade_in_progress = True

    # 거래 일시 중지 여부 확인
    with trading_pause_lock:
        if trading_paused:
            logger.info("거래가 일시 중지 상태입니다.")
            with trade_lock:
                trade_in_progress = False
            return

    try:
        # 현재 오더북 가져오기
        orderbook = fetch_orderbook(market)
        if not orderbook:
            logger.debug("오더북 가져오기 실패.")
            return

        bids, asks = process_orderbook(orderbook)
        if not bids or not asks:
            logger.debug("오더북 호가 처리 실패.")
            return

        # 매수량이 가장 많은 호가 찾기
        highest_bid = max(bids, key=lambda x: x['size'])
        buy_price = highest_bid['price'] + get_tick_size(highest_bid['price'])  # 한 틱 위에서 매수
        logger.debug(f"매수 가격 설정: {buy_price:.2f}원 (매수량: {highest_bid['size']} XRP)")

        # 매수 주문 걸기 전에 0.2초 대기 및 가격 변동 확인
        logger.debug(f"매수 주문 전 0.2초 대기 및 가격 변동 확인: 가격 {buy_price:.2f}원")
        time.sleep(0.2)
        orderbook_after = fetch_orderbook(market)
        if not orderbook_after:
            logger.debug("오더북 가져오기 실패. 매수 주문을 생성하지 않습니다.")
            return

        bids_after, asks_after = process_orderbook(orderbook_after)
        if not bids_after or not asks_after:
            logger.debug("오더북 호가 처리 실패. 매수 주문을 생성하지 않습니다.")
            return

        highest_bid_after = max(bids_after, key=lambda x: x['size'])
        if highest_bid_after['price'] != highest_bid['price']:
            logger.debug("매수 가격이 변동되어 매수 주문을 생성하지 않습니다.")
            return

        # 매수 주문 걸기
        buy_amount = 10000.0  # 예: 10,000원
        current_balance = get_current_krw_balance()

        # 남은 시드 금액 계산
        with totals_lock:
            remaining_seed = available_seed

        if remaining_seed <= 0:
            logger.info(f"시드 금액 {SEED_AMOUNT}원을 모두 투자했습니다. 추가 매수는 더 이상 진행되지 않습니다.")
            with trade_lock:
                trade_in_progress = False
            return

        # 매수 금액이 남은 시드 금액 및 잔고보다 큰 경우 조정
        buy_amount = min(buy_amount, remaining_seed, current_balance)

        if buy_amount < get_tick_size(buy_price):
            logger.debug(f"매수 금액이 최소 주문 금액보다 작아 매수 주문을 걸지 않습니다: {buy_amount:.2f}원")
            with trade_lock:
                trade_in_progress = False
            return

        volume = buy_amount / buy_price
        volume = round(volume, 6)  # 소수점 자리수 조정

        with open_order_lock():
            response_buy = place_order(market, 'bid', volume, buy_price)
            if response_buy and 'uuid' in response_buy:
                buy_order_id = response_buy['uuid']
                last_order_time = datetime.now(timezone.utc)  # 주문 시간 업데이트
                logger.debug(f"매수 주문 체결 대기: 매수 가격 {buy_price:.2f}원, 수량: {volume} XRP, 주문 ID {buy_order_id}")
            else:
                logger.debug(f"매수 주문 실행 실패 (매수 주문 ID 없음).")
                with trade_lock:
                    trade_in_progress = False

    except Exception as e:
        logger.debug(f"오더북 기반 매매 실행 중 오류 발생: {e}")
    finally:
        with trade_lock:
            trade_in_progress = False

# 실시간 주문 스케줄러 함수
def real_time_order_scheduler():
    while True:
        try:
            execute_orderbook_based_trading()
        except Exception as e:
            logger.debug(f"실시간 주문 스케줄링 중 오류 발생: {e}")
        time.sleep(0.2)  # 0.2초 간격으로 체크 (5번/초)

# 거래 일시 중지 및 재개 함수
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

# 메인 실행 함수
def main():
    logger.info("트레이딩 봇 시작")
    threading.Thread(target=real_time_order_scheduler, daemon=True).start()
    threading.Thread(target=monitor_buy_orders, daemon=True).start()
    threading.Thread(target=monitor_sell_orders, daemon=True).start()

    # 메인 스레드는 사용자 입력을 대기
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