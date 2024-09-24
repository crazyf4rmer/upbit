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
from collections import deque
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import threading

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

# 틱 사이즈 계산 함수
def get_tick_size(price):
    """
    가격에 따른 틱 사이즈를 동적으로 지정합니다.
    UXLINK의 가격대에 맞게 조정 필요.
    """
    if price < 0.01:
        return 0.00001
    elif price < 0.1:
        return 0.0001
    elif price < 1:
        return 0.001
    elif price < 10:
        return 0.01
    elif price < 100:
        return 0.1
    else:
        return 1.0

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
sell_orders = {}  # {buy_order_id: {'sell_order_id': sell_id, 'buy_price': buy_price, 'volume': volume}}
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
totals_lock = threading.Lock()  # 누적 변수 동기화를 위한 락

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
            if side == 'bid':
                # 매수 주문 개별 로그 제거
                # logger.info(f"매수 주문 대기: {float(price):.2f}원, 수량: {volume} UXLINK, 주문 ID: {order_id}")
                pass  # 로그 제거
            elif side == 'ask' and linked_order_id:
                sell_orders[linked_order_id] = {
                    'sell_order_id': order_id,
                    'buy_price': float(price),  # 매수 가격 저장
                    'volume': float(volume)      # 수량 저장
                }
                logger.debug(f"매도 주문 생성됨: {float(price):.2f}원, 수량: {volume} UXLINK, 매수 주문 ID: {linked_order_id}, 매도 주문 ID: {order_id}")
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

# 매도 주문 상태 확인 및 관리 함수
def monitor_sell_orders():
    global sell_orders, trade_session_counter, trade_in_progress, trading_paused
    while True:
        try:
            time.sleep(5)  # 5초마다 매도 주문 상태 확인
            with open_order_lock():
                for buy_id, sell_info in list(sell_orders.items()):
                    sell_id = sell_info['sell_order_id']
                    buy_price = sell_info['buy_price']
                    volume = sell_info['volume']
                    status = check_order_status(sell_id, 'ask')
                    if status:
                        state = status.get('state')
                        if state == 'done':
                            sell_price = float(status.get('price'))

                            # 매도 가격 검증
                            if sell_price <= buy_price:
                                logger.warning(
                                    f"매도 가격이 매수 가격보다 낮거나 같습니다: 매수 가격 {buy_price:.2f}원, 매도 가격 {sell_price:.2f}원 (주문 ID: {sell_id})"
                                )
                                # 추가 조치: 예를 들어, 자동으로 손절매 주문을 생성하거나 알림을 보낼 수 있습니다.

                            # 누적 변수 업데이트 및 로그 출력
                            with totals_lock:
                                total_buys += buy_price * volume
                                total_sells += sell_price * volume
                                fee_buy = buy_price * 0.0005  # 매수 수수료
                                fee_sell = sell_price * 0.0005  # 매도 수수료
                                cumulative_profit += (sell_price - buy_price - (fee_buy + fee_sell)) * volume

                                # 누적 요약 로그
                                trade_log = (
                                    "----------------------------------------------------------------------\n"
                                    f"{trade_session_counter}번 거래 세션에서\n"
                                    f"{buy_price:.2f}원 매수\n"
                                    f"{sell_price:.2f}원 매도 예정.\n"
                                    f"누적 매수 총액: {total_buys:.2f}원\n"
                                    f"누적 매도 총액: {total_sells:.2f}원\n"
                                    f"누적 순이익: {cumulative_profit:.2f}원\n"
                                    "----------------------------------------------------------------------"
                                )

                            logger.info(trade_log)

                            # 추적 중인 매도 주문에서 제거
                            del sell_orders[buy_id]
                            trade_session_counter += 1

                            # 거래 완료 시 플래그 해제
                            with trade_lock:
                                trade_in_progress = False

                        elif state in ['cancelled', 'failed']:
                            logger.info(f"매도 주문 취소됨: 주문 ID {sell_id}, 상태: {state}")
                            del sell_orders[buy_id]
                            # 거래 실패 시 플래그 해제
                            with trade_lock:
                                trade_in_progress = False
        except Exception as e:
            logger.debug(f"매도 주문 모니터링 중 오류 발생: {e}")

# 매도 주문 실행 함수 (매수 주문 체결 후)
def place_sell_order_on_buy(order_id, buy_price, volume):
    buy_fee_rate = 0.0005  # 매수 수수료 (0.05%)
    sell_fee_rate = 0.0005  # 매도 수수료 (0.05%)
    target_profit_rate = 0.002  # 목표 수익률 (0.2%)

    # 순수익률 보장을 위한 매도 가격 계산
    sell_price = (buy_price * (1 + buy_fee_rate) * (1 + target_profit_rate)) / (1 - sell_fee_rate)
    
    tick_size = get_tick_size(sell_price)
    # 소수점 자리수를 맞추기 위하여 올림 처리하여 수익 보장
    rounded_sell_price = math.ceil(sell_price / tick_size) * tick_size

    logger.debug(f"매도 가격 계산: {sell_price:.5f}원")
    logger.debug(f"반올림된 매도 가격: {rounded_sell_price:.2f}원 (틱 사이즈: {tick_size})")

    # 수익성 검증
    if not is_profit_possible(buy_price, rounded_sell_price):
        logger.debug(f"즉시 매도 수익 불가: 매수 가격 {buy_price}원, 매도 가격 {rounded_sell_price}원")
        with trade_lock:
            trade_in_progress = False
        return

    response_sell = place_order('KRW-UXLINK', 'ask', volume, rounded_sell_price, linked_order_id=order_id)

    if response_sell and 'uuid' in response_sell:
        sell_order_id = response_sell['uuid']
        logger.debug(f"매도 주문 실행: {rounded_sell_price:.2f}원, 수량: {volume} UXLINK (매수 주문 ID: {order_id}, 매도 주문 ID: {sell_order_id})")
    else:
        logger.debug(f"매도 주문 실행 실패: {rounded_sell_price:.2f}원, 수량: {volume} UXLINK (매수 주문 ID: {order_id})")
        with trade_lock:
            trade_in_progress = False

# 매수 주문 실행 함수 (실시간, 조건부)
def execute_buy_order_real_time():
    global last_order_time, trade_in_progress, trading_paused
    market = 'KRW-UXLINK'  # UXLINK 시장 심볼로 변경

    current_time = datetime.now(timezone.utc)
    time_since_last_order = (current_time - last_order_time).total_seconds()

    if time_since_last_order < 10:  # 최소 10초 대기
        logger.debug(f"주문 간 최소 10초 대기 중. 마지막 주문 이후 {time_since_last_order:.2f}초 경과.")
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
            logger.info("미체결 주문이 40개를 초과하여 거래가 일시 중지되었습니다.")
            trade_in_progress = False
            return

    try:
        orderbook = fetch_orderbook(market)
        if not orderbook:
            logger.debug("주문서 가져오기 실패.")
            return

        bids, asks = process_orderbook(orderbook)
        if not bids or not asks:
            logger.debug("주문서 처리 실패.")
            return

        best_bid = bids[0]['price']
        best_ask = asks[0]['price']
        mid_price = (best_bid + best_ask) / 2

        tick_size = get_tick_size(mid_price)
        decimal_places = len(str(tick_size).split('.')[-1]) if '.' in str(tick_size) else 0
        buy_price = round(mid_price - 0.3, decimal_places)
        logger.debug(f"계산된 매수 가격: {buy_price:.2f}원 (틱 사이즈: {tick_size})")

        volume = '7'  # 필요에 따라 조정 가능

        with open_order_lock():
            response_buy = place_order(market, 'bid', volume, buy_price)
            if response_buy and 'uuid' in response_buy:
                buy_order_id = response_buy['uuid']
                last_order_time = datetime.now(timezone.utc)  # 주문 시간 업데이트
                logger.debug(f"매수 주문 체결 대기: 매수 가격 {buy_price:.2f}원, 수량: {volume} UXLINK, 주문 ID: {buy_order_id}")
                threading.Thread(target=wait_and_place_sell_order, args=(buy_order_id, buy_price, volume), daemon=True).start()
            else:
                logger.debug(f"매수 주문 실행 실패 (매수 주문 ID 없음).")
    except Exception as e:
        logger.debug(f"매수 주문 실행 중 오류 발생: {e}")
    finally:
        with trade_lock:
            trade_in_progress = False

# 매수 주문 체결을 기다리고 매도 주문을 배치하는 함수
def wait_and_place_sell_order(buy_order_id, buy_price, volume, timeout=600, max_retries=3):
    """
    매수 주문이 체결될 때까지 대기한 후, 매도 주문을 배치합니다.
    timeout: 대기 시간(초) - 기본 10분
    max_retries: 최대 재시도 횟수
    """
    start_time = time.time()
    retries = 0
    while time.time() - start_time < timeout:
        status = check_order_status(buy_order_id, 'bid')
        if status and status.get('state') == 'done':
            # 실제 체결된 매수가격을 사용
            executed_buy_price = float(status.get('price', buy_price))
            current_time_str = time.strftime("%H:%M:%S", time.localtime())
            logger.debug(f"{current_time_str} - 매수 주문 체결됨: {executed_buy_price:.2f}원, 주문 ID: {buy_order_id}")
            # 매도 주문 배치
            place_sell_order_on_buy(buy_order_id, executed_buy_price, volume)
            return
        elif status and status.get('state') in ['cancelled', 'failed']:
            current_time_str = time.strftime("%H:%M:%S", time.localtime())
            logger.debug(f"{current_time_str} - 매수 주문 취소됨 또는 실패: 주문 ID {buy_order_id}, 상태: {status.get('state')}")
            with trade_lock:
                trade_in_progress = False
            return
        retries += 1
        if retries >= max_retries:
            logger.debug(f"매수 주문 체결 대기 중 최대 재시도 횟수에 도달함: 주문 ID {buy_order_id}")
            with trade_lock:
                trade_in_progress = False
            return
        time.sleep(2)  # 2초 간격으로 상태 확인
    logger.debug(f"매수 주문 체결 대기 시간 초과: 주문 ID {buy_order_id}")
    with trade_lock:
        trade_in_progress = False

# 실시간 주문 스케줄러 함수
def real_time_order_scheduler():
    while True:
        try:
            execute_buy_order_real_time()
        except Exception as e:
            logger.debug(f"실시간 주문 스케줄링 중 오류 발생: {e}")
        time.sleep(1)  # 1초 간격으로 체크

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

# 미체결 주문 관리 함수
def manage_open_orders():
    global trading_paused
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
                                logger.info(f"10분 이상 미체결 되어 취소된 매수 주문: 주문 ID {order_id}, 매수 가격: {order.get('price')}원, 수량: {order.get('volume')} UXLINK")
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

# 실시간 주문 스케줄러 시작과 매도 주문 모니터링, 미체결 주문 관리 스레드 시작
def start_threads():
    # 매도 주문 상태 모니터링 스레드 시작
    def run_monitor_sell_orders():
        while True:
            try:
                monitor_sell_orders()
            except Exception as e:
                logger.debug(f"매도 주문 모니터링 스레드 오류 발생: {e}")
                time.sleep(5)  # 잠시 대기 후 재시작

    monitor_sell_thread = threading.Thread(target=run_monitor_sell_orders, name="MonitorSellOrders")
    monitor_sell_thread.daemon = True
    monitor_sell_thread.start()
    logger.debug("매도 주문 모니터링 스레드가 시작되었습니다.")

    # 실시간 주문 스케줄러 스레드 시작
    def run_real_time_order_scheduler():
        while True:
            try:
                real_time_order_scheduler()
            except Exception as e:
                logger.debug(f"실시간 주문 스케줄러 스레드 오류 발생: {e}")
                time.sleep(5)  # 잠시 대기 후 재시작

    real_time_order_thread = threading.Thread(target=run_real_time_order_scheduler, name="RealTimeOrderScheduler")
    real_time_order_thread.daemon = True
    real_time_order_thread.start()
    logger.debug("실시간 주문 스케줄러 스레드가 시작되었습니다.")

    # 미체결 주문 관리 스레드 시작
    def run_manage_open_orders():
        while True:
            try:
                manage_open_orders()
            except Exception as e:
                logger.debug(f"미체결 주문 관리 스레드 오류 발생: {e}")
                time.sleep(5)  # 잠시 대기 후 재시작

    manage_orders_thread = threading.Thread(target=run_manage_open_orders, name="ManageOpenOrders")
    manage_orders_thread.daemon = True
    manage_orders_thread.start()
    logger.debug("미체결 주문 관리 스레드가 시작되었습니다.")

if __name__ == "__main__":
    start_threads()
    logger.info("트레이딩 봇이 조건부 매수 및 지정가 매도 전략으로 실행 중입니다. 종료하려면 Ctrl+C를 누르세요.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("사용자에 의해 트레이딩 봇이 중지되었습니다.")
    except Exception as e:
        logger.debug(f"메인 루프에서 오류 발생: {e}")