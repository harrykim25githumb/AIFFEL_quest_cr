
##### 모델배포개론 Day6. 인증 및 미지어 처리 기초 - Section 6 실습 #####

import requests


### 테스트 1: API Key 없이 요청

response = requests.post(
    "http://localhost:8000/predict/image",
    files={"file": ("test.png", b"fake image data", "image/png")},
    # headers 없음 → 인증 실패
)

print(f"상태 코드: {response.status_code}")   # 401
print(f"응답: {response.json()}")



### 테스트 2: 잘못된 키 -> 401

response = requests.post(
    "http://localhost:8000/predict/image",
    files={"file": ("test.png", b"fake image data", "image/png")},
    headers={"X-API-Key": "wrong-key"},                      # *your code* — 잘못된 키
)

print(f"\n상태 코드: {response.status_code}")   # 401
print(f"응답: {response.json()}")
 


### 테스트 3: 올바른 키 + MNIST 이미지

with open("app/image_api.py", "r", encoding="utf-8") as f:
    content = f.read()

# 응답 '계약'의 키만 label로 바꾼다 — 내부 키(model_utils의 predicted_class)는 그대로!
# (파일 전체를 무차별 replace하면 result["predicted_class"] 조회까지 깨져 KeyError가 난다)
old = '"predicted_class": result["predicted_class"]'
new = '"label": result["predicted_class"]'
if old in content:
    with open("app/image_api.py", "w", encoding="utf-8") as f:
        f.write(content.replace(old, new))
    print("\n✅ 응답 키를 label로 변경 (내부 키는 그대로)")
else:
    print("✅ 이미 수정되어 있음")


with open("app/image_api.py", "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if "label" in line.lower() or "predict" in line.lower():
            print(f"{i}: {line.rstrip()}")



from torchvision import datasets
from PIL import Image
import io
import requests

# MNIST 테스트 이미지 가져오기
test_dataset = datasets.MNIST(root="data", train=False, download=True)
test_image, test_label = test_dataset[0]   # 첫 번째 테스트 이미지

# PIL 이미지 → bytes 변환
buf = io.BytesIO()
test_image.save(buf, format="PNG")
image_bytes = buf.getvalue()

print(f"\n테스트 이미지 정답: {test_label}")

# API 호출
response = requests.post(
    "http://localhost:8000/predict/image",
    files={"file": ("digit.png", image_bytes, "image/png")},
    headers={"X-API-Key": "test-key-001"},                   # *your code* — 올바른 키
)

print(f"상태 코드: {response.status_code}")   # 200
result = response.json()
print(f"예측 결과: {result}")




### 테스트 4: 잘못된 파일 형식 -> 400

response = requests.post(
    "http://localhost:8000/predict/image",
    files={"file": ("test.txt", b"this is not an image", "text/plain")},
    headers={"X-API-Key": "test-key-001"},
)

print(f"\n상태 코드: {response.status_code}")   # 400
print(f"응답: {response.json()}")



### 테스트 5: 여러 이미지 연속 테스트

import requests
from torchvision import datasets
from PIL import Image
import io

test_dataset = datasets.MNIST(root="data", train=False, download=True)

print("\n=== 연속 추론 테스트 (5장) ===\n")

for i in range(5):
    img, label = test_dataset[i]

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    resp = requests.post(
        "http://localhost:8000/predict/image",
        files={"file": (f"digit_{i}.png", buf.getvalue(), "image/png")},
        headers={"X-API-Key": "test-key-001"},
    )

    r = resp.json()
    predicted = r.get("label", "?")
    confidence = r.get("confidence", 0)
    match = "✅" if str(label) == str(predicted) else "❌"

    print(f"  이미지 {i}: 정답={label}, 예측={predicted}, 확신도={confidence:.4f} {match}")





