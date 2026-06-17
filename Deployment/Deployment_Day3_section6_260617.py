
##### 모델배포실습 Day3 - Section6 실습 코드 ######


import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchvision import datasets
import time

test_dataset = datasets.MNIST(root="data", train=False, download=True)

def concurrent_pixel_test(n_requests=3):
    """실제 모델 추론으로 동시 요청을 테스트합니다."""
    def send(i):
        image, label = test_dataset[i % len(test_dataset)]
        pixels = (np.array(image) / 255.0).tolist()
        start = time.time()
        resp = requests.post(
            "http://localhost:8000/predict/pixels",
            json={"pixels": pixels},
            timeout=30,
        )
        return {
            "id": i + 1,
            "elapsed": round(time.time() - start, 2),
            "status": resp.status_code,
        }

    print(f"\n{'='*50}")
    print(f"  {n_requests}개 동시 요청 (실제 추론)")
    print(f"{'='*50}")

    start = time.time()
    with ThreadPoolExecutor(max_workers=n_requests) as ex:
        futures = [ex.submit(send, i) for i in range(n_requests)]
        results = [f.result() for f in as_completed(futures)]
    total = round(time.time() - start, 2)

    for r in sorted(results, key=lambda x: x["id"]):
        print(f"  요청 #{r['id']}: {r['elapsed']}초 (HTTP {r['status']})")
    print(f"  전체: {total}초")



# 동시 요청 수를 늘려가며 테스트
for n in [1, 2, 4, 8]:
    concurrent_pixel_test(n)
    time.sleep(1)



##### 에러 핸들링 테스트 #####

print("=" * 50)
print("  에러 핸들링 테스트")
print("=" * 50)

# 정상 요청
image, label = test_dataset[0]
pixels = (np.array(image) / 255.0).tolist()
resp = requests.post("http://localhost:8000/predict/pixels", json={"pixels": pixels})

print(resp.status_code)
print(resp.json())
#  print(f"\n[정상 요청] 상태: {resp.status_code}, 예측: {resp.json()['predicted_class']}")

data = resp.json()
if resp.status_code == 200:
    print(f"\n[정상 요청] 상태: {resp.status_code}, 예측: {data['predicted_class']}")
else:
    print(f"\n[오류] 상태: {resp.status_code}")
    print(data)


# 잘못된 픽셀 크기
resp = requests.post(
    "http://localhost:8000/predict/pixels",
    json={"pixels": [[0.0] * 14 for _ in range(14)]}
)
print(f"[잘못된 크기] 상태: {resp.status_code}")

# 잘못된 Base64
resp = requests.post(
    "http://localhost:8000/predict/image",
    json={"image_base64": "not_valid!!!"}
)
print(f"[잘못된 Base64] 상태: {resp.status_code}, 에러: {resp.json().get('detail', 'N/A')}")

# 헬스체크
resp = requests.get("http://localhost:8000/health")
print(f"[헬스체크] 상태: {resp.status_code}, 응답: {resp.json()}")

