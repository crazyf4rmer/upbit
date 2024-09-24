import requests
import jwt
import uuid
import hashlib
import time
from urllib.parse import urlencode

# Upbit API Keys 설정 (사용자 본인의 키로 대체하세요)
access_key = 'Dxt6TllIfUOSCExhxXg2AsFUTJjvnhFIt2HhEVqc'
secret_key = 'Xm1Xyxd2UA7703nJa503yxM9J0cVNlBpfxjoijAe'
server_url = 'https://api.upbit.com'

# 틱 사이즈 계산 함수
def get_tick_size(price):
    return 0.1
    # if price < 10:
    #     return 0.01
    # elif price < 100:
    #     return 0.1
    # elif price < 1000:
    #     return 1.0
    # elif price < 10000:
    #     return 5.0
    # elif price < 100000:
    #     return 10.0
    # elif price < 500000:
    #     return 50.0
    # else:
    #     return 100.0

# 주문서 조회 함수
def get_orderbook(market='KRW-TRX'):
    url = "https://api.upbit.com/v1/orderbook"
    params = {'markets': market}
    response = requests.get(url, params=params)
    return response.json()

# 매수 주문 정보 계산 함수
def get_buy_order_info(bids):
    best_bid_price = bids[0]['price']
    tick_size = get_tick_size(best_bid_price)
    cum_bid_volume_krw = 0
    largest_price = 0
    for bid in bids[1:]:  # 최우선 매수 호가 밑의 호가들
        price = bid['price']
        size = bid['size']
        volume_krw = price * size
        cum_bid_volume_krw += volume_krw
        largest_price = max(largest_price, price)
        if cum_bid_volume_krw >= 100000000:  # 1억 원 초과 시
            break
    if cum_bid_volume_krw >= 100000000:
        buy_price = largest_price + tick_size
        return buy_price
    else:
        return None  # 조건 미충족

# 매도 주문 정보 계산 함수
def get_sell_order_info(asks):
    best_ask_price = asks[0]['price']
    tick_size = get_tick_size(best_ask_price)
    cum_ask_volume_krw = 0
    smallest_price = 0
    for ask in asks[1:]:  # 최우선 매도 호가 위의 호가들
        price = ask['price']
        size = ask['size']
        volume_krw = price * size
        cum_ask_volume_krw += volume_krw
        if smallest_price == 0:
            smallest_price = price
        else:
            smallest_price = min(smallest_price, price)
        if cum_ask_volume_krw >= 100000000:  # 1억 원 초과 시
            break
    if cum_ask_volume_krw >= 100000000:
        sell_price = smallest_price - tick_size
        return sell_price
    else:
        return None  # 조건 미충족

# 매수/매도 주문 실행 함수
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

    jwt_token = jwt.encode(payload, secret_key)
    authorize_token = 'Bearer {}'.format(jwt_token)
    headers = {"Authorization": authorize_token}

    response = requests.post(server_url + '/v1/orders', params=query, headers=headers)

    return response.json()

# 주문서 데이터 처리 함수
def process_orderbook(orderbook):
    orderbook_units = orderbook[0]['orderbook_units']
    bids = []
    asks = []
    for unit in orderbook_units:
        bids.append({'price': unit['bid_price'], 'size': unit['bid_size']})
        asks.append({'price': unit['ask_price'], 'size': unit['ask_size']})
    return bids, asks

# 메인 실행 함수
def main():
    while True:
        try:
            orderbook = get_orderbook()
            bids, asks = process_orderbook(orderbook)

            # 매수 조건 확인 및 주문
            buy_price = get_buy_order_info(bids)
            if buy_price is not None:
                print(buy_price)
                side = 'bid'  # 매수
                volume = 4500  # 매수 수량 설정 (예: 1000 TRX)
                price = buy_price
                response = place_order('KRW-TRX', side, volume, price)
                print('매수 주문 실행:', response)

            # 매도 조건 확인 및 주문
            sell_price = get_sell_order_info(asks)
            if sell_price is not None:
                side = 'ask'  # 매도
                volume = 4500  # 매도 수량 설정 (예: 1000 TRX)
                price = sell_price
                response = place_order('KRW-TRX', side, volume, price)
                print('매도 주문 실행:', response)

            # 다음 조회 전 대기 (API 호출 제한을 준수하기 위해)
            time.sleep(5)  # 1초 대기

        except Exception as e:
            print('에러 발생:', e)
            time.sleep(1)

if __name__ == "__main__":
    main()