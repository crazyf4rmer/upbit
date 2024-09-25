import os
import requests
import jwt
import uuid
import hashlib
import time
import logging
from urllib.parse import urlencode, unquote
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import threading
from decimal import Decimal, ROUND_UP
from logging.handlers import RotatingFileHandler

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
SEED_AMOUNT = 100000.0  # 단위: KRW

# 수익률 설정 (0.5%)
PROFIT_TARGET_RATE = 0.005  # 0.5% 상승 시

# 그리드 메이킹 관련 설정 (마켓 메이킹 전략 도입)
GRID_LEVELS = 5  # 매수/매도 주문 레벨 수
GRID_SPREAD_MIN = 0.3  # 최소 그리드 간격 비율 (%)
GRID_SPREAD_MAX = 1.0  # 최대 그리드 간격 비율 (%)

# 기타 설정
STOP_LOSS_START = 3  # 손절 시작 비율 (%)
STOP_LOSS_STEP = 1    # 손절 단계 비율 (%)
STOP_LOSS_MAX = 10    # 최대 손절 비율 (%)
ADDITIONAL_STOP_LOSS_RATE = 0.03  # 추가: 3% 하락 시 손절

# 최소 주문 금액 및 수량 설정
MIN_BUY_AMOUNT_KRW = 5000  # 매수 최소 주문 금액 (예: 5,000 KRW)
MIN_SELL_VOLUME = 0.001     # 매도 최소 수량 (예: 0.001 SEI)

# 틱 사이즈 계산 함수 (Tick Size를 0.1 KRW로 설정)
def get_tick_size(price):
    """
    가격에 따른 틱 사이즈를 지정합니다.
    Upbit의 SEI 틱 사이즈에 맞게 조정되었습니다.
    """
    if price < 100:
        return 0.01
    elif 100 <= price < 1000:
        return 0.1
    elif 1000 <= price < 10000:
        return 1
    elif 10000 <= price < 100000:
        return 10
    elif 100000 <= price < 500000:
        return 50
    elif 500000 <= price < 1000000:
        return 100
    elif 1000000 <= price < 2000000:
        return 500
    else:
        return 1000

# 가격 반올림을 위한 함수 추가
def round_price(price, tick_size):
    """
    주어진 가격을 틱 사이즈에 맞게 올림 처리하여 반올림합니다.
    """
    price_decimal = Decimal(str(price))
    tick_size_decimal = Decimal(str(tick_size))
    rounded = (price_decimal / tick_size_decimal).to_integral_value(rounding=ROUND_UP) * tick_size_decimal
    return float(rounded)

# 수익성 검증 함수
def is_profit_possible(buy_price, sell_price, fee_rate=0.001):
    """
    수수료를 고려하여 매도 가격이 수익을 낼 수 있는지 확인합니다.
    fee_rate: 매수 + 매도 수수료 합산 (0.1% = 0.0005 * 2)
    """
    total_fee_buy = buy_price * 0.0005  # 매수 수수료 (0.05%)
    total_fee_sell = sell_price * 0.0005  # 매도 수수료 (0.05%)
    total_cost = buy_price + total_fee_buy
    total_revenue = sell_price - total_fee_sell
    profit = total_revenue - total_cost
    logger.debug(f"Buy Price: {buy_price}, Sell Price: {sell_price}, Profit: {profit}")
    return profit > 0

# 전역 변수 및 데이터 구조
sell_orders = {}  # {buy_order_id: {'sell_order_id': sell_id, 'buy_price': buy_price, 'volume': volume, 'target_price': target_price, 'created_at': datetime}}
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
totals_lock = threading.Lock()  # 누적 변수 동기화를 위한 락

# 마지막 매도 체결 시간 추적 변수 및 락
last_sell_fill_time = datetime.min.replace(tzinfo=timezone.utc)
last_sell_fill_time_lock = threading.Lock()

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

# 현재 SEI 잔액을 조회하는 함수 추가
def get_current_sei_balance():
    """
    Upbit API의 /v1/accounts 엔드포인트를 활용하여 현재 SEI 잔액을 조회합니다.
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
            if account.get('currency') == 'SEI':
                balance = float(account.get('balance', 0))
                logger.debug(f"현재 SEI 잔고: {balance} SEI")
                return balance
        return 0.0
    except requests.exceptions.Timeout:
        logger.debug("잔고 조회 요청 타임아웃")
    except Exception as e:
        logger.debug(f"잔고 조회 중 오류 발생: {e}")
        return 0.0

# 평균 매수가 추적 변수 추가
total_position = {
    'total_volume': 0.0,
    'total_cost': 0.0,
    'average_price': 0.0
}
position_lock = threading.Lock()

# 매수/매도 주문 실행 함수
def place_order(market, side, volume, price, linked_order_id=None, ord_type='limit'):
    global sell_orders  # 전역 변수 사용
    params = {
        'market': market,
        'side': side,
        'volume': str(volume),  # 문자열로 변환
        'ord_type': ord_type,
    }

    # 가격이 None인 경우 키를 제거하여 시장가 주문으로 설정
    if price is not None:
        params['price'] = str(price)  # 문자열로 변환

    if linked_order_id:
        params['identifier'] = linked_order_id  # 링크된 주문 ID 사용

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
        logger.debug(f"JWT 인코딩 오류: {e}")
        return None

    authorization = f'Bearer {jwt_token}'
    headers = {
        "Authorization": authorization
    }

    # POST 요청 전송 (json=params으로 전송)
    try:
        response = requests.post(f"{server_url}/v1/orders", json=params, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        order_id = data.get('uuid')
        if order_id:
            current_time = datetime.now(timezone.utc)
            if side == 'bid':
                # 매수 주문 체결 대기 (total_position 업데이트는 체결 시점에 수행)
                logger.debug(f"매수 주문 생성됨: 가격 {price:.2f}원, 수량: {volume} SEI, 주문 ID: {order_id}")
            elif side == 'ask' and linked_order_id:
                sell_orders[linked_order_id] = {
                    'sell_order_id': order_id,
                    'buy_price': float(price),  # 매수 가격 저장
                    'volume': float(volume),    # 수량 저장
                    'target_price': float(price) * (1 + PROFIT_TARGET_RATE),  # 목표 가격 설정
                    'created_at': current_time  # 매도 주문 생성 시간 추가
                }
                logger.debug(f"매도 주문 생성됨: 가격 {float(price) * (1 + PROFIT_TARGET_RATE):.2f}원, 수량: {volume} SEI, 매수 주문 ID: {linked_order_id}, 매도 주문 ID: {order_id}, 목표 가격: {float(price) * (1 + PROFIT_TARGET_RATE):.2f}원")
            elif side == 'ask' and not linked_order_id:
                # 일반 매도 주문 (재분배 시 사용)
                sell_orders[f"general_{order_id}"] = {
                    'sell_order_id': order_id,
                    'buy_price': 0.0,  # 일반 매도는 매수 가격이 없음
                    'volume': float(volume),
                    'target_price': float(price),
                    'created_at': current_time
                }
                logger.debug(f"일반 매도 주문 생성됨: 가격 {price:.2f}원, 수량: {volume} SEI, 주문 ID {order_id}")
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
def place_sell_order_on_buy(order_id, buy_price, volume, new_sell_price=None):
    global trade_in_progress  # 전역 변수 사용
    buy_fee_rate = 0.0005  # 매수 수수료 (0.05%)
    sell_fee_rate = 0.0005  # 매도 수수료 (0.05%)
    target_profit_rate = PROFIT_TARGET_RATE  # 목표 수익률

    if new_sell_price:
        sell_price = new_sell_price
    else:
        # 목표 수익률에 따른 매도 가격 계산
        sell_price = buy_price * (1 + target_profit_rate)

    tick_size = get_tick_size(sell_price)
    # 소수점 자리수를 맞추기 위하여 올림 처리하여 수익 보장
    rounded_sell_price = round_price(sell_price, tick_size)

    if not new_sell_price:
        logger.debug(f"매도 가격 계산: {sell_price:.2f}원")
        logger.debug(f"반올림된 매도 가격: {rounded_sell_price:.2f}원 (틱 사이즈: {tick_size})")

    # 수익성 검증
    if not is_profit_possible(buy_price, rounded_sell_price):
        logger.debug(f"즉시 매도 수익 불가: 매수 가격 {buy_price}원, 매도 가격 {rounded_sell_price}원")
        with trade_lock:
            trade_in_progress = False
        return None

    response_sell = place_order('KRW-SEI', 'ask', volume, rounded_sell_price, linked_order_id=order_id)

    if response_sell and 'uuid' in response_sell:
        sell_order_id = response_sell['uuid']
        if new_sell_price:
            # 재매도 주문일 경우 특별히 로그를 남김
            logger.debug(f"재매도 주문 실행됨: {rounded_sell_price:.2f}원, 수량: {volume} SEI (매수 주문 ID: {order_id}, 재매도 주문 ID: {sell_order_id})")
        else:
            logger.debug(f"매도 주문 실행됨: {rounded_sell_price:.2f}원, 수량: {volume} SEI (매수 주문 ID: {order_id}, 매도 주문 ID: {sell_order_id})")
        return response_sell
    else:
        logger.debug(f"매도 주문 실행 실패: {rounded_sell_price:.2f}원, 수량: {volume} SEI (매수 주문 ID: {order_id})")
        with trade_lock:
            trade_in_progress = False

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

# 매수 주문이 체결되면 매도 주문을 생성하는 함수
def wait_and_place_sell_order(buy_order_id, buy_price, volume):
    """
    매수 주문이 체결되면 매도 주문을 생성하는 함수
    """
    global trade_in_progress
    while True:
        status = check_order_status(buy_order_id, 'bid')
        if status:
            state = status.get('state')
            if state == 'done':
                # 매도 주문 실행
                response_sell = place_sell_order_on_buy(buy_order_id, buy_price, volume)
                if response_sell:
                    logger.info(f"매도 주문이 성공적으로 생성되었습니다: 주문 ID {response_sell.get('uuid')}")
                    
                    # 매도 주문이 체결될 때까지 기다림 (monitor_sell_orders에서 처리)
                    
                    # 매수 주문이 체결되었으므로 total_position 업데이트
                    with position_lock:
                        total_position['total_volume'] += float(volume)
                        total_position['total_cost'] += float(buy_price) * float(volume)
                        if total_position['total_volume'] > 0:
                            total_position['average_price'] = total_position['total_cost'] / total_position['total_volume']
                        else:
                            total_position['average_price'] = 0
                        logger.debug(f"매수 주문 체결: 가격 {buy_price:.2f}원, 수량: {volume} SEI, 평균 매수가: {total_position['average_price']:.2f}원")
                else:
                    logger.debug(f"매도 주문 생성 실패: 매수 주문 ID {buy_order_id}")
                break
            elif state in ['cancelled', 'failed']:
                logger.info(f"매수 주문이 취소되거나 실패했습니다: 주문 ID {buy_order_id}, 상태: {state}")
                with trade_lock:
                    trade_in_progress = False
                break
        time.sleep(5)  # 5초 간격으로 상태 확인

# 마켓 메이킹 매수 주문 배치 함수 추가
def place_grid_buy_orders(base_price, tick_size, levels):
    global total_invested  # 전역 변수 사용
    buy_prices = [base_price - (i * tick_size) for i in range(1, levels + 1)]
    for price in buy_prices:
        # 기존 주문과 중복되지 않도록 확인
        existing_buy_prices = {float(order.get('price')) for order in get_open_orders() if order.get('side') == 'bid'}
        if price in existing_buy_prices:
            logger.debug(f"이미 매수 주문이 걸려있는 가격: {price:.2f}원")
            continue

        # 매수 시 사용할 금액 설정 (예: 시드 금액의 일정 비율)
        buy_amount = SEED_AMOUNT / (levels * 2)  # 시드 금액을 레벨 수의 두 배로 나눠 각 주문에 배정
        current_balance = get_current_krw_balance()

        # 남은 시드 금액 계산
        remaining_seed = SEED_AMOUNT - total_invested
        if remaining_seed <= 0:
            logger.info(f"시드 금액 {SEED_AMOUNT}원을 모두 투자했습니다. 추가 매수는 더 이상 진행되지 않습니다.")
            break

        # 매수 금액이 남은 시드 금액과 현재 잔고를 초과하지 않도록 조정
        adjusted_buy_amount = min(buy_amount, remaining_seed, current_balance)

        # 최소 주문 금액 검증
        if adjusted_buy_amount < MIN_BUY_AMOUNT_KRW:
            logger.debug(f"매수 금액 {adjusted_buy_amount:.2f} KRW이 최소 주문 금액 {MIN_BUY_AMOUNT_KRW} KRW을 충족하지 못해서 매수 주문을 걸지 않습니다.")
            continue

        volume = adjusted_buy_amount / price
        volume = round(volume, 6)  # 소수점 자리수 조정

        with open_order_lock():
            response_buy = place_order('KRW-SEI', 'bid', volume, price)
            if response_buy and 'uuid' in response_buy:
                buy_order_id = response_buy['uuid']
                logger.debug(f"매수 주문 체결 대기: 매수 가격 {price:.2f}원, 수량: {volume} SEI, 주문 ID {buy_order_id}")
                threading.Thread(target=wait_and_place_sell_order, args=(buy_order_id, price, volume), daemon=True).start()
                # 투자 금액 누적
                with totals_lock:
                    total_invested += adjusted_buy_amount
            else:
                logger.debug(f"매수 주문 실행 실패 (매수 주문 ID 없음).")

# 마켓 메이킹 매도 주문 배치 함수 추가
def place_grid_sell_orders(base_price, tick_size, levels):
    """
    마켓 메이킹을 위한 그리드 매도 주문을 배치하는 함수
    """
    sell_prices = [base_price + (i * tick_size) for i in range(1, levels + 1)]
    current_sei_balance = get_current_sei_balance()

    for price in sell_prices:
        # 기존 주문과 중복되지 않도록 확인
        existing_sell_prices = {float(order.get('price')) for order in get_open_orders() if order.get('side') == 'ask'}
        if price in existing_sell_prices:
            logger.debug(f"이미 매도 주문이 걸려있는 가격: {price:.2f}원")
            continue

        # 매도 시 물량 설정 (보유 SEI의 일정 비율)
        sell_volume = current_sei_balance / (levels * 2)  # 보유 SEI를 레벨 수의 두 배로 나눠 각 주문에 배정
        sell_volume = round(sell_volume, 6)

        # 최소 매도 수량 검증
        if sell_volume < MIN_SELL_VOLUME:
            logger.debug(f"매도 물량 {sell_volume} SEI가 최소 매도 수량 {MIN_SELL_VOLUME} SEI을 충족하지 못해 매도 주문을 걸지 않습니다.")
            continue

        # 매도 금액 검증
        total_sell_amount = price * sell_volume
        if total_sell_amount < MIN_BUY_AMOUNT_KRW:
            logger.debug(f"매도 금액 {total_sell_amount:.2f} KRW이 최소 주문 금액 {MIN_BUY_AMOUNT_KRW} KRW을 충족하지 않아서 매도 주문을 걸지 않습니다.")
            continue

        with open_order_lock():
            response_sell = place_order('KRW-SEI', 'ask', sell_volume, price, linked_order_id=None, ord_type='limit')
            if response_sell and 'uuid' in response_sell:
                sell_order_id = response_sell['uuid']
                logger.debug(f"매도 주문 생성됨: {price:.2f}원, 수량: {sell_volume} SEI, 주문 ID {sell_order_id}")
            else:
                logger.debug(f"매도 주문 실행 실패 (매도 주문 ID 없음).")

# 전체 매도 주문을 지정가로 매도하고 모니터링하는 함수
def take_profit_and_monitor():
    """
    설정한 수익률에 도달했을 때 기존 매도 주문을 취소하고, 지정가 매도 주문을 걸어두는 함수
    """
    global total_position, sell_orders
    with position_lock:
        average_price = total_position['average_price']
        total_volume = total_position['total_volume']

    if total_volume == 0:
        # 보유 물량이 없으면 종료
        return

    # 0.5% 수익 실현 가격 계산
    take_profit_price = average_price * (1 + PROFIT_TARGET_RATE)

    logger.info(f"수익 실현을 위해 모든 매도 주문을 취소하고, 가격 {take_profit_price:.2f}원으로 지정가 매도 주문을 걸어둡니다.")

    # 모든 기존 매도 주문 취소
    cancel_all_sell_orders()

    # 현재 보유 SEI 잔량 조회
    current_sei_balance = get_current_sei_balance()
    if current_sei_balance < MIN_SELL_VOLUME:
        logger.warning(f"보유 SEI 수량 {current_sei_balance} SEI이 최소 매도 수량 {MIN_SELL_VOLUME} SEI을 충족하지 못합니다.")
        return

    # 지정가 매도 주문 생성
    response_sell = place_order('KRW-SEI', 'ask', current_sei_balance, take_profit_price, ord_type='limit')

    if response_sell and 'uuid' in response_sell:
        sell_order_id = response_sell['uuid']
        sell_orders[f"profit_sell_{sell_order_id}"] = {
            'sell_order_id': sell_order_id,
            'buy_price': average_price,
            'volume': current_sei_balance,
            'target_price': take_profit_price,
            'created_at': datetime.now(timezone.utc)
        }
        logger.info(f"수익 실현 매도 주문 생성됨: 주문 ID {sell_order_id}, 가격: {take_profit_price:.2f}원, 수량: {current_sei_balance} SEI")
    else:
        logger.warning("수익 실현 매도 주문 생성 실패.")

# 주문서를 취소하고 매도 주문을 재분배하는 함수
def redistribute_sell_orders(buy_id, original_sell_price, volume):
    """
    매도 주문 가격이 설정된 비율 이상 하락 시, 매도 주문을 재분배하는 함수
    """
    current_market_price = get_current_market_price('KRW-SEI')
    if current_market_price is None:
        logger.warning("현재 시장 가격을 가져올 수 없어 매도 주문 재분배를 진행할 수 없습니다.")
        return

    # 새로운 매도 주문 가격을 현재 시장 가격으로 설정
    target_price = current_market_price

    logger.info(f"현재 시장 가격이 매도 가격보다 {ADDITIONAL_STOP_LOSS_RATE*100}% 이상 하락하였습니다: {current_market_price:.2f}원 <= {original_sell_price:.2f}원")
    logger.info(f"매도 주문을 재분배합니다. 새로운 매도 가격: {target_price:.2f}원")

    # 기존 매도 주문 취소
    sell_info = sell_orders.get(buy_id)
    if sell_info:
        sell_order_id = sell_info['sell_order_id']
        cancel_response = cancel_order(sell_order_id)
        if cancel_response and 'uuid' in cancel_response:
            logger.info(f"매도 주문 취소됨: 주문 ID {sell_order_id}")
            del sell_orders[buy_id]
        else:
            logger.warning(f"매도 주문 취소 실패: 주문 ID {sell_order_id}")

    # 새로운 매도 주문 분배 (예: 3개의 주문으로 분할)
    num_new_orders = 3
    spread_percentage = 0.03  # 3% 범위

    for i in range(1, num_new_orders + 1):
        # 각 주문의 가격 설정 (예: 균등 분할)
        new_price = target_price * (1 + (-spread_percentage/2) + (spread_percentage/(num_new_orders - 1)) * (i - 1))

        # 매도 주문 금액 검증
        total_sell_amount = new_price * (volume / num_new_orders)
        if total_sell_amount < MIN_BUY_AMOUNT_KRW:
            logger.warning(f"재분배 매도 주문 금액 {total_sell_amount:.2f} KRW이 최소 주문 금액 {MIN_BUY_AMOUNT_KRW} KRW을 충족하지 못합니다. 주문을 걸지 않습니다.")
            continue

        # 매도 주문 수량
        sell_volume = volume / num_new_orders
        sell_volume = round(sell_volume, 6)

        # 가격을 tick size에 맞게 반올림
        sell_price_rounded = round_price(new_price, get_tick_size(new_price))

        # 매도 주문 생성
        response_sell = place_order('KRW-SEI', 'ask', sell_volume, sell_price_rounded, linked_order_id=buy_id, ord_type='limit')

        if response_sell and 'uuid' in response_sell:
            new_sell_order_id = response_sell['uuid']
            sell_orders[buy_id] = {
                'sell_order_id': new_sell_order_id,
                'buy_price': total_position['average_price'],
                'volume': sell_volume,
                'target_price': sell_price_rounded,
                'created_at': datetime.now(timezone.utc)
            }
            logger.info(f"재분배 매도 주문 생성됨: 주문 ID {new_sell_order_id}, 가격: {sell_price_rounded:.2f}원, 수량: {sell_volume} SEI")
        else:
            logger.warning(f"재분배 매도 주문 생성 실패: 가격: {sell_price_rounded:.2f}원, 수량: {sell_volume} SEI")

# 주문 상태를 지속적으로 모니터링하고 관리하는 함수
def monitor_sell_orders():
    global sell_orders, trade_session_counter, trade_in_progress, trading_paused, total_buys, total_sells, cumulative_profit, total_invested, last_sell_fill_time
    while True:
        try:
            time.sleep(5)  # 5초마다 매도 주문 상태 확인
            with open_order_lock():
                for buy_id, sell_info in list(sell_orders.items()):
                    sell_id = sell_info['sell_order_id']
                    buy_price = sell_info['buy_price']
                    volume = sell_info['volume']
                    target_price = sell_info.get('target_price', buy_price * PROFIT_TARGET_RATE)
                    status = check_order_status(sell_id, 'ask')
                    if status:
                        state = status.get('state')
                        if state == 'done':
                            sell_price = float(status.get('price'))

                            # 누적 변수 업데이트 및 로그 출력
                            with totals_lock:
                                total_buys += buy_price * volume
                                total_sells += sell_price * volume
                                fee_buy = buy_price * 0.0005  # 매수 수수료
                                fee_sell = sell_price * 0.0005  # 매도 수수료
                                profit = (sell_price - buy_price - (fee_buy + fee_sell)) * volume
                                cumulative_profit += profit
                                
                                # **매도 완료 시 투자 금액 복구**
                                total_invested -= (buy_price * volume)
                                if total_invested < 0:
                                    logger.warning(f"총 투자 금액이 음수가 되었습니다: {total_invested}. 0으로 초기화합니다.")
                                    total_invested = 0.0

                                # 누적 요약 로그
                                trade_log = (
                                    "----------------------------------------------------------------------\n"
                                    f"{trade_session_counter}번 거래 세션에서\n"
                                    f"{buy_price:.2f}원 매수\n"
                                    f"{sell_price:.2f}원 매도 완료.\n"
                                    f"누적 매수 총액: {total_buys:.2f}원\n"
                                    f"누적 매도 총액: {total_sells:.2f}원\n"
                                    f"누적 순이익: {cumulative_profit:.2f}원\n"
                                    f"현재 사용 가능한 시드: {SEED_AMOUNT - total_invested:.2f}원\n"
                                    "----------------------------------------------------------------------"
                                )

                            logger.info(trade_log)

                            # 마지막 매도 체결 시간 업데이트
                            with last_sell_fill_time_lock:
                                last_sell_fill_time = datetime.now(timezone.utc)

                            # 추적 중인 매도 주문에서 제거
                            del sell_orders[buy_id]
                            trade_session_counter += 1

                            # 거래 완료 시 플래그 해제
                            with trade_lock:
                                trade_in_progress = False

                        elif state in ['wait', 'open']:
                            # 목표 가격 도달 여부 확인
                            current_market_price = get_current_market_price('KRW-SEI')
                            if current_market_price and current_market_price >= target_price:
                                new_sell_price = current_market_price  # 현재 시장 가격으로 설정
                                tick_size = get_tick_size(new_sell_price)
                                rounded_new_sell_price = round_price(new_sell_price, tick_size)

                                logger.debug(f"목표 수익률 도달 시 매도 주문 조정: {rounded_new_sell_price:.2f}원 (틱 사이즈: {tick_size})")

                                # 매도 주문 재조정을 위해 매도 주문 취소 및 재등록
                                cancel_order(sell_id)
                                response_new_sell = place_sell_order_on_buy(buy_id, buy_price, volume, new_sell_price=rounded_new_sell_price)
                                if response_new_sell and 'uuid' in response_new_sell:
                                    new_sell_order_id = response_new_sell['uuid']
                                    sell_orders[buy_id]['sell_order_id'] = new_sell_order_id
                                    sell_orders[buy_id]['target_price'] = rounded_new_sell_price
                                    sell_orders[buy_id]['created_at'] = datetime.now(timezone.utc)
                                    logger.info(f"목표 수익률 도달에 따른 매도 주문 생성됨: {rounded_new_sell_price:.2f}원, 주문 ID: {new_sell_order_id}")
                                else:
                                    logger.warning(f"매도 주문 재설정 실패: {rounded_new_sell_price:.2f}원, 수량: {volume} SEI, 매수 주문 ID: {buy_id}")

                        elif state in ['cancelled', 'failed']:
                            logger.info(f"매도 주문 취소됨: 주문 ID {sell_id}, 상태: {state}")
                            del sell_orders[buy_id]
                            # 거래 실패 시 플래그 해제
                            with trade_lock:
                                trade_in_progress = False
        except Exception as e:
            logger.debug(f"매도 주문 모니터링 중 오류 발생: {e}")

# 손절 및 수익 실현 관리 함수 추가
def monitor_average_price_and_take_profit_and_stop_loss():
    """
    평균 매수가를 모니터링하고, 수익 실현 및 손절 조건을 관리하는 함수
    """
    while True:
        try:
            with position_lock:
                average_price = total_position['average_price']
                total_volume = total_position['total_volume']

            if total_volume == 0:
                time.sleep(10)
                continue

            current_price = get_current_market_price('KRW-SEI')
            if current_price is None:
                logger.debug("현재 시장 가격을 가져올 수 없습니다.")
                time.sleep(10)
                continue

            loss_percentage = ((average_price - current_price) / average_price) * 100
            profit_percentage = ((current_price - average_price) / average_price) * 100

            logger.debug(f"평균 매수가: {average_price:.2f}원, 현재 가격: {current_price:.2f}원, 손실 비율: {loss_percentage:.2f}%, 수익 비율: {profit_percentage:.2f}%")

            # 수익 실현 로직
            if profit_percentage >= (PROFIT_TARGET_RATE * 100):
                logger.info(f"수익 목표 도달: 현재 가격이 평균 매수가의 {PROFIT_TARGET_RATE*100}% 이상입니다. 모든 매도 주문을 취소하고 지정가 매도 주문을 생성합니다.")
                take_profit_and_monitor()

            # 손절 로직
            if loss_percentage >= STOP_LOSS_START:
                # 손절 단계 계산
                loss_step = int((loss_percentage - STOP_LOSS_START) // STOP_LOSS_STEP) + 1
                current_stop_loss = STOP_LOSS_START + (STOP_LOSS_STEP * loss_step)

                if current_stop_loss > STOP_LOSS_MAX:
                    current_stop_loss = STOP_LOSS_MAX

                target_loss_price = average_price * (1 - (current_stop_loss / 100))

                # 해당 손실 비율에 해당하는 매도 주문 찾기
                amount_to_sell = (total_invested * (current_stop_loss / 100))
                amount_sold = 0.0

                with open_order_lock():
                    # 매도 주문을 가격이 낮은 순서대로(가장 낮은 가격 주문 먼저) 처리
                    sorted_sell_orders = sorted(sell_orders.items(), key=lambda x: x[1]['target_price'])  # 가격 기준 정렬

                    for buy_id, sell_info in sorted_sell_orders:
                        if amount_sold >= amount_to_sell:
                            break
                        sell_id = sell_info['sell_order_id']
                        sell_price = sell_info['target_price']
                        volume = sell_info['volume']
                        # 목표 손실 가격 이하인 매도 주문 취소
                        if sell_price <= target_loss_price:
                            cancel_response = cancel_order(sell_id)
                            if cancel_response and 'uuid' in cancel_response:
                                logger.info(f"손절을 위해 매도 주문 취소됨: 주문 ID {sell_id}, 가격: {sell_price:.2f}원, 수량: {volume} SEI")
                                # 해당 매도 주문을 시장가로 매도
                                market_sell_response = place_order('KRW-SEI', 'ask', volume, None, linked_order_id=None, ord_type='price')
                                if market_sell_response and 'uuid' in market_sell_response:
                                    logger.info(f"시장가 매도 주문 생성됨: 주문 ID {market_sell_response.get('uuid')}, 수량: {volume} SEI")
                                    # 누적 손실 계산
                                    amount_sold += (average_price - current_price) * volume
                                    # 기존 sell_orders에서 제거
                                    del sell_orders[buy_id]
                                else:
                                    logger.warning(f"시장가 매도 주문 생성 실패: 주문 ID {sell_id}, 수량: {volume} SEI")
                    logger.debug(f"손절을 위해 매도하려는 총 금액: {amount_to_sell}원, 실제 매도된 금액: {amount_sold}원")

            # 3% 이상의 하락 시 매도 주문 재분배
            for buy_id, sell_info in list(sell_orders.items()):
                sell_price = sell_info['target_price']
                # 현재 시장 가격이 매도 가격보다 3% 이상 하락했는지 확인
                if current_price <= sell_price * (1 - ADDITIONAL_STOP_LOSS_RATE):
                    logger.info(f"현재 가격이 매도 가격보다 {ADDITIONAL_STOP_LOSS_RATE*100}% 이상 하락하였습니다: {current_price:.2f}원 <= {sell_price:.2f}원")
                    redistribute_sell_orders(buy_id, sell_price, sell_info['volume'])

            time.sleep(30)  # 30초 간격으로 모니터링
        except Exception as e:
            logger.debug(f"손절 및 수익 실현 관리 중 오류 발생: {e}")
            time.sleep(30)

# 전체 매도 주문을 취소하는 함수
def cancel_all_sell_orders():
    """
    현재 모든 지정가 매도 주문을 취소하는 함수
    """
    open_orders = get_open_orders()
    for order in open_orders:
        if order.get('side') == 'ask':
            order_id = order.get('uuid')
            if order_id:
                cancel_response = cancel_order(order_id)
                if cancel_response and 'uuid' in cancel_response:
                    logger.info(f"매도 주문 취소됨: 주문 ID {order_id}")
                    # 추적 중인 sell_orders에서 제거
                    for buy_id, sell_info in list(sell_orders.items()):
                        if sell_info['sell_order_id'] == order_id:
                            del sell_orders[buy_id]
                else:
                    logger.warning(f"매도 주문 취소 실패: 주문 ID {order_id}")

# 미체결 주문 관리 함수
def manage_open_orders():
    global trading_paused  # 전역 변수 사용
    while True:
        try:
            time.sleep(60)  # 60초마다 체크
            open_orders = get_open_orders()
            open_order_count = len(open_orders)
            logger.debug(f"현재 미체결 주문 수: {open_order_count}")

            with trading_pause_lock:
                if not trading_paused and open_order_count > 40:
                    trading_paused = True
                    logger.info(f"미체결 주문 수가 40개를 초과하여 거래가 일시 중지되었습니다. 현재 미체결 주문 수: {open_order_count}")
                elif trading_paused and open_order_count <= 40:
                    trading_paused = False
                    logger.info(f"미체결 주문 수가 40개 이하로 감소하여 거래가 재개되었습니다. 현재 미체결 주문 수: {open_order_count}")

            # 10분 이상 미체결인 매수 주문 취소
            current_time = datetime.now(timezone.utc)
            for order in open_orders:
                if order.get('side') == 'bid':
                    created_at_str = order.get('created_at')
                    if not created_at_str:
                        continue
                    created_at = parse_created_at(created_at_str)
                    if not created_at:
                        continue
                    elapsed = current_time - created_at
                    if elapsed > timedelta(minutes=10):
                        order_id = order.get('uuid')
                        if order_id:
                            cancel_response = cancel_order(order_id)
                            if cancel_response and 'uuid' in cancel_response:
                                logger.info(f"10분 이상 미체결 되어 취소된 매수 주문: 주문 ID {order_id}, 매수 가격: {order.get('price')}원, 수량: {order.get('volume')} SEI")
                                # 투자 금액 복구
                                buy_price = float(order.get('price', 0))
                                volume = float(order.get('volume', 0))
                                buy_amount = buy_price * volume
                                with totals_lock:
                                    global total_invested
                                    total_invested -= buy_amount
                                    if total_invested < 0:
                                        logger.warning(f"총 투자 금액이 음수가 되었습니다: {total_invested}. 0으로 초기화합니다.")
                                        total_invested = 0.0
                            else:
                                logger.info(f"매수 주문 취소 실패: 주문 ID {order_id}")

            # 기존 미체결 주문 관리 로직 유지 (예: 50개 초과 시 취소)
            if open_order_count > 50:
                excess_orders = open_order_count - 50
                logger.info(f"미체결 주문이 50개를 초과했습니다. {excess_orders}개의 주문을 취소합니다.")

                # 미체결 주문을 시간 순으로 정렬 (가장 최근 주문부터 취소)
                # Upbit API는 주문 생성 시간을 반환하지 않을 수 있으므로, 주문 목록을 역순으로 처리
                # 실제로는 주문 생성 시간을 기준으로 정렬하는 것이 좋습니다.
                for order in reversed(open_orders):
                    if excess_orders <= 0:
                        break
                    order_id = order.get('uuid')
                    if order_id:
                        cancel_response = cancel_order(order_id)
                        if cancel_response and 'uuid' in cancel_response:
                            logger.info(f"미체결 주문 취소됨: 주문 ID {order_id}")
                            excess_orders -= 1
                        else:
                            logger.info(f"미체결 주문 취소 실패: 주문 ID {order_id}")
        except Exception as e:
            logger.debug(f"미체결 주문 관리 중 오류 발생: {e}")

# 잔고 동기화 스레드 추가
def balance_sync_thread():
    while True:
        try:
            current_balance = get_current_krw_balance()
            current_sei_balance = get_current_sei_balance()
            logger.debug(f"잔고 동기화 - 현재 잔고: {current_balance:.2f} KRW, {current_sei_balance:.6f} SEI")
            # 필요한 경우, 로컬 변수나 기타 상태를 업데이트
            # 예를 들어, 거래 일시 중지 여부를 업데이트할 수도 있습니다
        except Exception as e:
            logger.debug(f"잔고 동기화 중 오류 발생: {e}")
        time.sleep(30)  # 30초마다 잔고 동기화

# 실시간 주문 스케줄러 함수 추가
def real_time_order_scheduler():
    """
    스케줄러 함수: 현재 시장 가격을 기반으로 그리드 매수 및 매도 주문을 배치합니다.
    """
    market = 'KRW-SEI'
    while True:
        try:
            current_price = get_current_market_price(market)
            if current_price is None:
                logger.debug("현재 시장 가격을 가져올 수 없어 주문을 배치할 수 없습니다.")
            else:
                tick_size = get_tick_size(current_price)
                logger.debug(f"실시간 주문 스케줄러: 현재 가격 {current_price:.2f}원, 틱 사이즈 {tick_size}")
                
                # 그리드 매수 주문 배치
                place_grid_buy_orders(base_price=current_price, tick_size=tick_size, levels=GRID_LEVELS)
                
                # 그리드 매도 주문 배치
                place_grid_sell_orders(base_price=current_price, tick_size=tick_size, levels=GRID_LEVELS)
                
            time.sleep(60)  # 주문 배치 주기 설정 (예: 60초)
        except Exception as e:
            logger.debug(f"실시간 주문 스케줄러 오류 발생: {e}")
            time.sleep(60)  # 오류 발생 시 대기 후 재시도

# 미체결 주문 관리 함수 추가
def cancel_unfilled_sells_if_no_fills():
    """
    3분 간격으로 매도 주문이 하나도 체결되지 않았다면, 가장 오래된 미체결 매도 주문 2개(가능하면)를 취소하고 일반 매매를 재개합니다.
    또는
    만약 지정가 매수 주문이 없을 경우, 가장 멀리 있는 미체결 매도 주문 2개(가능하면)를 취소하고 해당 주문금액만큼 시장가로 던져서 매도한 뒤 일반 매매를 지속합니다.
    """
    market = 'KRW-SEI'
    check_interval = 180  # 3분 (초 단위)

    while True:
        try:
            time.sleep(check_interval)  # 3분 대기

            with last_sell_fill_time_lock:
                time_since_last_sell_fill = datetime.now(timezone.utc) - last_sell_fill_time

            if time_since_last_sell_fill < timedelta(minutes=3):
                # 최근 3분 이내에 매도 주문이 체결되었으므로 취소하지 않음
                logger.debug("최근 3분 내에 매도 주문이 체결되었습니다. 취소 작업을 건너뜁니다.")
                continue

            with open_order_lock():
                # 현재 미체결 매수 주문 목록 가져오기
                open_buy_orders = [order for order in get_open_orders() if order.get('side') == 'bid']

                if open_buy_orders:
                    # 매수 주문이 있는 경우, 가장 오래된 2개 취소
                    open_buy_orders_sorted = sorted(open_buy_orders, key=lambda x: parse_created_at(x.get('created_at')), reverse=False)
                    orders_to_cancel = open_buy_orders_sorted[:2]

                    for order in orders_to_cancel:
                        order_id = order.get('uuid')
                        price = float(order.get('price', 0))
                        volume = float(order.get('volume', 0))
                        if order_id:
                            logger.info(f"체결되지 않은 매수 주문을 취소합니다: 주문 ID {order_id}, 가격: {price:.2f}원, 수량: {volume} SEI")
                            cancel_response = cancel_order(order_id)
                            if cancel_response and 'uuid' in cancel_response:
                                logger.info(f"매수 주문 취소됨: 주문 ID {order_id}")
                                # 투자 금액 복구
                                buy_amount = price * volume
                                with totals_lock:
                                    global total_invested
                                    total_invested -= buy_amount
                                    if total_invested < 0:
                                        logger.warning(f"총 투자 금액이 음수가 되었습니다: {total_invested}. 0으로 초기화합니다.")
                                        total_invested = 0.0
                            else:
                                logger.warning(f"매수 주문 취소 실패: 주문 ID {order_id}")
                else:
                    # 매수 주문이 없는 경우, 가장 멀리 있는 매도 주문 2개 취소 후 시장가 매도
                    if not sell_orders:
                        logger.debug("취소할 매도 주문이 없습니다.")
                        continue

                    # 가장 멀리 있는 (가격 기준) 매도 주문 2개 선택
                    open_sell_orders_sorted = sorted(sell_orders.items(), key=lambda x: x[1]['target_price'], reverse=True)
                    orders_to_cancel = open_sell_orders_sorted[:2]

                    for buy_id, sell_info in orders_to_cancel:
                        sell_id = sell_info['sell_order_id']
                        sell_price = sell_info['target_price']
                        volume = sell_info['volume']
                        if sell_id:
                            logger.info(f"체결되지 않은 매도 주문을 취소하고 시장가로 매도합니다: 주문 ID {sell_id}, 가격: {sell_price:.2f}원, 수량: {volume} SEI")
                            cancel_response = cancel_order(sell_id)
                            if cancel_response and 'uuid' in cancel_response:
                                logger.info(f"매도 주문 취소됨: 주문 ID {sell_id}")
                                # 시장가 매도 주문 실행
                                market_sell_response = place_order('KRW-SEI', 'ask', volume, None, linked_order_id=None, ord_type='price')
                                if market_sell_response and 'uuid' in market_sell_response:
                                    market_sell_id = market_sell_response['uuid']
                                    logger.info(f"시장가 매도 주문 생성됨: 주문 ID {market_sell_id}, 수량: {volume} SEI")
                                    # sell_orders에서 제거
                                    del sell_orders[buy_id]
                                else:
                                    logger.warning(f"시장가 매도 주문 생성 실패: 주문 ID {sell_id}, 수량: {volume} SEI")
                            else:
                                logger.warning(f"매도 주문 취소 실패: 주문 ID {sell_id}")
        except Exception as e:
            logger.debug(f"미체결 매도 주문 취소 중 오류 발생: {e}")

# 실시간 주문 스케줄러와 매도 주문 모니터링, 미체결 주문 관리, 잔고 동기화, 손절 및 수익 실현 관리, 미체결 매도 주문 취소 스레드를 시작하는 함수
def start_threads():
    # 매도 주문 상태 모니터링 스레드 시작
    monitor_sell_thread = threading.Thread(target=monitor_sell_orders, name="MonitorSellOrders")
    monitor_sell_thread.daemon = True
    monitor_sell_thread.start()
    logger.debug("매도 주문 모니터링 스레드가 시작되었습니다.")

    # 실시간 주문 스케줄러 스레드 시작
    real_time_order_thread = threading.Thread(target=real_time_order_scheduler, name="RealTimeOrderScheduler")
    real_time_order_thread.daemon = True
    real_time_order_thread.start()
    logger.debug("실시간 주문 스케줄러 스레드가 시작되었습니다.")

    # 미체결 주문 관리 스레드 시작
    manage_orders_thread = threading.Thread(target=manage_open_orders, name="ManageOpenOrders")
    manage_orders_thread.daemon = True
    manage_orders_thread.start()
    logger.debug("미체결 주문 관리 스레드가 시작되었습니다.")

    # 잔고 동기화 스레드 시작
    balance_sync = threading.Thread(target=balance_sync_thread, name="BalanceSync")
    balance_sync.daemon = True
    balance_sync.start()
    logger.debug("잔고 동기화 스레드가 시작되었습니다.")

    # 손절 및 수익 실현 관리 스레드 시작
    stop_loss_take_profit_thread = threading.Thread(target=monitor_average_price_and_take_profit_and_stop_loss, name="StopLossTakeProfitMonitor")
    stop_loss_take_profit_thread.daemon = True
    stop_loss_take_profit_thread.start()
    logger.debug("손절 및 수익 실현 관리 스레드가 시작되었습니다.")

    # 미체결 매도 주문 취소 스레드 시작
    cancel_unfilled_sell_thread = threading.Thread(target=cancel_unfilled_sells_if_no_fills, name="CancelUnfilledSellsIfNoFills")
    cancel_unfilled_sell_thread.daemon = True
    cancel_unfilled_sell_thread.start()
    logger.debug("미체결 매도 주문 취소 스레드가 시작되었습니다.")

if __name__ == "__main__":
    start_threads()
    logger.info("트레이딩 봇이 SEI를 대상으로 일반 거래를 유지하며, 시드머니의 0.5% 수익률 달성 시 지정가 매도하도록 실행 중입니다. 종료하려면 Ctrl+C를 누르세요.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("사용자에 의해 트레이딩 봇이 중지되었습니다.")
    except Exception as e:
        logger.debug(f"메인 루프에서 오류 발생: {e}")