import os
import requests
import jwt
import uuid
import hashlib
import time
import logging
import json
from urllib.parse import urlencode
from dotenv import load_dotenv
from websocket import WebSocketApp
import threading
from urllib.parse import urlencode, unquote

# .env 파일 로드 (스크립트 파일의 동일 디렉토리에 위치)
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

# Upbit API Keys 설정 (환경 변수에서 불러오기)
access_key = os.getenv('UPBIT_ACCESS_KEY')
secret_key = os.getenv('UPBIT_SECRET_KEY')
server_url = 'https://api.upbit.com'

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,  # 초기 디버깅을 위해 DEBUG로 설정
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# 틱 사이즈 계산 함수
def get_tick_size(price):
    return 0.1
    # 필요 시 다양한 가격대에 따른 틱 사이즈를 적용할 수 있습니다.

# 수익성 검증 함수
def is_profit_possible(buy_price, sell_price, fee_rate=0.001):
    """
    수수료를 고려하여 매도 가격이 수익을 낼 수 있는지 확인합니다.
    fee_rate: 매수 + 매도 수수료 (0.1% = 0.001)
    """
    total_fee_buy = buy_price * 0.0005
    total_fee_sell = sell_price * 0.0005
    total_cost = buy_price + total_fee_buy
    total_revenue = sell_price - total_fee_sell
    profit = total_revenue - total_cost
    return profit > 0

# 매수 주문 정보 계산 함수
def get_buy_order_info(bids, volume_threshold):
    if not bids:
        return None
    best_bid_price = bids[0]['price']
    tick_size = get_tick_size(best_bid_price)
    cum_bid_volume_krw = 0
    largest_price = 0
    for bid in bids[1:]:
        try:
            price = float(bid['price'])
            size = float(bid['size'])
            volume_krw = price * size
            cum_bid_volume_krw += volume_krw
            largest_price = max(largest_price, price)
            if cum_bid_volume_krw >= volume_threshold:
                break
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Error processing bid: {bid}, Error: {e}")
            continue
    if cum_bid_volume_krw >= volume_threshold:
        buy_price = round(largest_price + tick_size, 1)
        return buy_price
    else:
        return None

# 매도 주문 정보 계산 함수
def get_sell_order_info(asks, volume_threshold):
    if not asks:
        return None
    best_ask_price = asks[0]['price']
    tick_size = get_tick_size(best_ask_price)
    cum_ask_volume_krw = 0
    smallest_price = None
    for ask in asks[1:]:
        try:
            price = float(ask['price'])
            size = float(ask['size'])
            volume_krw = price * size
            cum_ask_volume_krw += volume_krw
            if smallest_price is None:
                smallest_price = price
            else:
                smallest_price = min(smallest_price, price)
            if cum_ask_volume_krw >= volume_threshold:
                break
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Error processing ask: {ask}, Error: {e}")
            continue
    if cum_ask_volume_krw >= volume_threshold and smallest_price is not None:
        sell_price = round(smallest_price - tick_size, 1)
        return sell_price
    else:
        return None

# 매수/매도 주문 실행 함수 (JSON 형식으로 수정됨)
def place_order(market, side, volume, price, ord_type='limit'):
    query = {
        'market': market,
        'side': side,
        'volume': volume,  
        'price': price,    
        'ord_type': ord_type,
    }

    # JSON 문자열 생성 (키를 정렬하고 공백 제거)
    json_body = json.dumps(query, separators=(',', ':'), sort_keys=True)
    # json_body = unquote(urlencode(query, doseq=True)).encode("utf-8")
    m = hashlib.sha512()
    m.update(json_body.encode('utf-8'))
    query_hash = m.hexdigest()

    payload = {
        'access_key': access_key,
        'nonce': str(uuid.uuid4()),
        'query_hash': query_hash,
        'query_hash_alg': 'SHA512',
    }

    try:
        # JWT 토큰 생성 (HS256 알고리즘 사용)
        jwt_token = jwt.encode(payload, secret_key, algorithm='HS256')

        # PyJWT 버전별 처리 (토큰이 bytes일 경우 디코딩 필요)
        if isinstance(jwt_token, bytes):
            jwt_token = jwt_token.decode('utf-8')

        authorize_token = f'Bearer {jwt_token}'
        headers = {
            "Authorization": authorize_token,
            "Content-Type": "application/json; charset=utf-8"  # JSON 형식으로 변경
        }

        # 디버깅 정보 로그
        logger.debug(f"Headers: {headers}")
        logger.debug(f"Body: {json_body}")

        # POST 요청 전송 (JSON 데이터로 전송)
        response = requests.post(server_url + '/v1/orders', data=json_body, headers=headers)

        # 응답 상태 코드 로그
        logger.debug(f"Response Status Code: {response.status_code}")

        # 상태 코드 검사
        response.raise_for_status()

        try:
            data = response.json()
            # 응답 로그
            logger.debug(f"Response Body: {data}")
            return data
        except json.JSONDecodeError:
            # JSON 파싱 실패 시
            logger.error(f"Failed to parse JSON response: {response.text}")
            return None

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err} - Response: {response.text}")
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        # 디버깅을 위해 추가 정보 로그
        logger.debug(f"Query string: {json_body}")
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
        logger.error(f"Error processing orderbook data: {e}")
        return None, None

# 동적 볼륨 임계값 계산 함수
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
            logger.error(f"Unexpected trades format: {trades}")
            return 100000000  # 기본값 설정 (예: 1억 원)
    except Exception as e:
        logger.error(f"Error fetching dynamic volume threshold: {e}")
        return 100000000  # 기본값 설정 (예: 1억 원)

# WebSocket 메시지 처리 함수
def on_message(ws, message):
    try:
        logger.debug(f"Received WebSocket message: {message}")  # 추가된 로그
        orderbook = json.loads(message)
        bids, asks = process_orderbook(orderbook)
        if bids is None or asks is None:
            return

        # 동적 볼륨 임계값 계산
        volume_threshold = get_dynamic_volume_threshold()
        logger.info(f"동적 볼륨 임계값: {volume_threshold}")

        # 매수 조건 확인 및 주문
        buy_price = get_buy_order_info(bids, volume_threshold)
        if buy_price is not None:
            logger.info(f"매수 가격: {buy_price}")
            # 수수료를 고려하여 매도 가격 계산
            required_sell_price = buy_price * 1.001  # 매수 가격의 0.1% 이상
            # 매도 호가 중 조건을 만족하는 호가 찾기
            for ask in asks:
                ask_price = ask['price']
                if ask_price >= required_sell_price and is_profit_possible(buy_price, ask_price):
                    sell_price = round(ask_price, 1)
                    logger.info(f"매도 가격: {sell_price}")

                    # 매수 주문 실행
                    side_buy = 'bid'  # 매수
                    volume = '4500'  # 매수 수량 설정 (예: 4500 TRX)
                    response_buy = place_order('KRW-TRX', side_buy, volume, str(buy_price))
                    if response_buy:
                        logger.info(f'매수 주문 실행: {response_buy}')
                        # 매도 주문 실행
                        side_sell = 'ask'  # 매도
                        response_sell = place_order('KRW-TRX', side_sell, volume, str(sell_price))
                        if response_sell:
                            logger.info(f'매도 주문 실행: {response_sell}')
                        else:
                            logger.error('매도 주문에 실패했습니다.')
                    else:
                        logger.error('매수 주문에 실패했습니다.')
                    break  # 한 번의 매수/매도 후 루프 종료

    except Exception as e:
        logger.error(f"Error processing message: {e}")

def on_error(ws, error):
    logger.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    logger.info("WebSocket closed, attempting to reconnect in 5 seconds...")
    time.sleep(5)
    start_websocket()

def on_open(ws):
    # 구독할 마켓과 채널 설정
    subscribe_data = [
        {"ticket": "test"},
        {"type": "orderbook", "codes": ["KRW-TRX"], "isOnlyRealtime": True}
    ]
    ws.send(json.dumps(subscribe_data))
    logger.info("WebSocket connection opened and subscription message sent")

def start_websocket():
    websocket_url = "wss://api.upbit.com/websocket/v1"
    ws = WebSocketApp(
        websocket_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    # WebSocket을 별도의 스레드에서 실행
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()

    logger.info("WebSocket thread started.")

    return ws

if __name__ == "__main__":
    ws_instance = start_websocket()
    logger.info("Trading bot is running. Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
        ws_instance.close()