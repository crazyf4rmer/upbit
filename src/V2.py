import jwt
import hashlib
import os
import requests
import uuid
import json
from urllib.parse import urlencode
from dotenv import load_dotenv


# .env 파일 로드
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)


# 환경 변수에서 API 키와 서버 URL 가져오기
access_key = os.getenv('UPBIT_OPEN_API_ACCESS_KEY')
secret_key = os.getenv('UPBIT_OPEN_API_SECRET_KEY')
server_url = os.getenv('UPBIT_OPEN_API_SERVER_URL')


print(access_key, secret_key, server_url)
# 주문 파라미터 설정
params = {
    'market': 'KRW-BTC',
    'side': 'bid',
    'ord_type': 'limit',
    'price': '100.0',
    'volume': '0.01'
}

# JSON 직렬화 (공백 없이)
json_params = json.dumps(params, separators=(',', ':'), ensure_ascii=False)

# query_hash 생성
query_hash = hashlib.sha512(json_params.encode('utf-8')).hexdigest()

# JWT 페이로드 구성
payload = {
    'access_key': access_key,
    'nonce': str(uuid.uuid4()),
    'query_hash': query_hash,
    'query_hash_alg': 'SHA512',
}

# JWT 토큰 생성 (HS512 알고리즘 사용)
jwt_token = jwt.encode(payload, secret_key, algorithm='HS256')
if isinstance(jwt_token, bytes):
    jwt_token = jwt_token.decode('utf-8')

# 인증 헤더 설정
authorization = f'Bearer {jwt_token}'
headers = {
    'Authorization': authorization,
    'Content-Type': 'application/json',
}

# POST 요청 전송
res = requests.post(server_url + '/v1/orders', data=json_params, headers=headers)

# 응답 출력
print(res.json())
