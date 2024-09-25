import requests
import time
from datetime import datetime, timedelta

# 상수 설정
TARGET_VOLUME = 100_000_000_000  # 100억 원
FLUCTUATION_PCT = 0.2  # ±0.2%

def get_krw_markets():
    """
    Upbit의 KRW 마켓에 있는 모든 코인의 목록을 가져옵니다.
    """
    url = "https://api.upbit.com/v1/market/all"
    params = {"isDetails": "false"}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        krw_markets = [item['market'] for item in data if item['market'].startswith('KRW-')]
        return krw_markets
    except Exception as e:
        print(f"KRW 마켓 코인 목록 가져오기 실패: {e}")
        return []

def get_latest_candle(market, unit=1):
    """
    지정된 코인과 시간 단위의 최신 캔들 데이터를 가져옵니다.
    """
    url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    # UTC 시간으로 변환
    to_time = datetime.utcnow().replace(second=0, microsecond=0)
    params = {
        "market": market,
        "count": 1,
        "to": to_time.strftime('%Y-%m-%dT%H:%M:00Z')
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        else:
            return None
    except Exception as e:
        print(f"{market}의 캔들 데이터 가져오기 실패: {e}")
        return None

def main():
    krw_markets = get_krw_markets()
    if not krw_markets:
        print("KRW 마켓 코인 목록을 가져오지 못했습니다. 스크립트를 종료합니다.")
        return
    print(f"총 KRW 마켓 코인 개수: {len(krw_markets)}")

    while True:
        # 현재 UTC 시간 기준으로 다음 1분까지 대기
        now = datetime.utcnow()
        next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
        sleep_seconds = (next_minute - now).total_seconds()
        print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} UTC: 다음 분석까지 {sleep_seconds:.2f}초 대기 중...")
        time.sleep(sleep_seconds)

        # 대상 1분
        target_time = next_minute - timedelta(minutes=1)
        target_time_str = target_time.strftime('%Y-%m-%d %H:%M')
        print(f"\n=== {target_time_str} UTC 기간의 데이터 분석 시작 ===")

        results = []
        for idx, market in enumerate(krw_markets, 1):
            candle = get_latest_candle(market)
            if candle is None:
                continue

            # 거래대금 계산
            acc_trade_price = float(candle.get('candle_acc_trade_price', 0))
            if acc_trade_price < TARGET_VOLUME:
                continue  # 거래대금이 기준 미만인 코인 제외

            # 시가, 고가, 저가 추출
            open_price = float(candle.get('opening_price', 0))
            high_price = float(candle.get('high_price', 0))
            low_price = float(candle.get('low_price', 0))

            # 변동 횟수 계산 (횡보: 변동 없음)
            is_sideways = True
            if high_price >= open_price * (1 + FLUCTUATION_PCT / 100) or low_price <= open_price * (1 - FLUCTUATION_PCT / 100):
                is_sideways = False

            if is_sideways:
                results.append({
                    'market': market,
                    'acc_trade_price': acc_trade_price,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                })

            # 진행 표시
            if idx % 50 == 0 or idx == len(krw_markets):
                print(f"진행: {idx}/{len(krw_markets)} 코인 분석 완료")

            # API 요청 간격 조절 (속도 조절을 위해 약간의 대기)
            time.sleep(0.05)  # 초당 약 20 요청

        # 결과 정렬 (거래대금 내림차순)
        sorted_results = sorted(
            results,
            key=lambda x: x['acc_trade_price'],
            reverse=True
        )

        # 정렬된 결과 출력
        print(f"\n=== {target_time_str} UTC 기간 분석 결과 ===")
        if sorted_results:
            print(f"{'순위':<5} {'코인':<10} {'거래대금(KRW)':<15}")
            for rank, item in enumerate(sorted_results, 1):
                print(f"{rank:<5} {item['market']:<10} {item['acc_trade_price']:<15,.0f}")
        else:
            print("횡보하는 코인이 없습니다.")
        print("==============================================\n")

if __name__ == "__main__":
    main()