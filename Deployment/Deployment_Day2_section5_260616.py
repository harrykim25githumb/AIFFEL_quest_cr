

# import torch
# import torch.nn as nn
# from torchvision import transforms


# # ===== 모델 정의 =====
# class SimpleClassifier(nn.Module):
#     """
#     간단한 이미지 분류 모델
#     - 입력: 1x28x28 (MNIST와 동일한 크기)
#     - 출력: 10개 클래스에 대한 확률
#     """
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * 7 * 7, 128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, num_classes),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x


# # ===== 전처리 파이프라인 =====
# preprocess = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((28, 28)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,)),
# ])


# # ===== 모델 로드 =====
# def load_model(model_path: str = "models/mnist_state_dict.pth") -> nn.Module:
#     """
#     저장된 state_dict를 로드하여 추론 가능한 모델을 반환합니다.
#     """
#     model = SimpleClassifier(num_classes=10)
#     model.load_state_dict(
#         torch.load(model_path, map_location="cpu", weights_only=True)
#     )
#     model.eval()
#     return model


# # ===== 추론 함수 =====
# def predict(model: nn.Module, input_tensor: torch.Tensor) -> dict:
#     """
#     모델에 입력 텐서를 전달하고 예측 결과를 반환합니다.

#     Args:
#         model: 로드된 PyTorch 모델
#         input_tensor: 전처리된 입력 텐서 (1, 1, 28, 28)

#     Returns:
#         dict: {"label": int, "confidence": float, "probabilities": list}
#     """
#     with torch.no_grad():
#         output = model(input_tensor)
#         probabilities = torch.softmax(output, dim=1)
#         confidence, predicted = torch.max(probabilities, dim=1)

#     return {
#         "label": predicted.item(),
#         "confidence": round(confidence.item(), 4),
#         "probabilities": probabilities[0].tolist(),
#     }



# # 정상 작동 여부 확인
# from app.model_utils import load_model, predict, preprocess
# print("✅ model_utils import 성공")

# # 모델 로드 테스트
# model = load_model("models/mnist_state_dict.pth")
# print(f"✅ 모델 로드 성공: {type(model).__name__}")




##### 테스트 1: 헬스 체크 #####

import requests

response = requests.get("http://localhost:8000/health")
print(f"상태 코드: {response.status_code}")
print(f"응답: {response.json()}")




##### 테스트2: MNIST 이미지로 추론 #####

from torchvision import datasets, transforms

# MNIST 테스트 데이터 로드
test_dataset = datasets.MNIST(
    root="data", train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
)

# 첫 번째 테스트 이미지 가져오기
test_image, true_label = test_dataset[0]
print(f"이미지 크기: {test_image.shape}")     # torch.Size([1, 28, 28])
print(f"정답 레이블: {true_label}")

# 픽셀 값을 리스트로 변환 (API에 보낼 형식)
pixel_values = test_image.flatten().tolist()
print(f"픽셀 값 개수: {len(pixel_values)}")   # 784



##### 실제 API 호출 #####

import json

# 추론 요청
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "pixel_values": pixel_values,
        "return_probabilities": False,
    }
)

print(f"상태 코드: {response.status_code}")
print(f"응답:")
print(json.dumps(response.json(), indent=2, ensure_ascii=False))



##### 테스트3 : 확률분포 요청 #####

# return_probabilities를 True로 설정
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "pixel_values": pixel_values,
        "return_probabilities": True,
    }
)

result = response.json()
print(f"예측: {result['label']} (확신도: {result['confidence']})")
print(f"\n클래스별 확률:")
for i, prob in enumerate(result['probabilities']):
    bar = "█" * int(prob * 50)
    print(f"  {i}: {prob:.4f} {bar}")



##### 테스트4: 여러 이미지 테스트 #####

# 10개 이미지를 연속으로 테스트
print(f"{'이미지':<8} {'정답':<6} {'예측':<6} {'확신도':<10} {'결과'}")
print("-" * 45)

correct = 0
for i in range(10):
    image, true_label = test_dataset[i]
    pixel_values = image.flatten().tolist()

    response = requests.post(
        "http://localhost:8000/predict",
        json={"pixel_values": pixel_values}
    )
    result = response.json()

    is_correct = result["label"] == true_label
    if is_correct:
        correct += 1

    mark = "✅" if is_correct else "❌"
    print(f"  #{i:<5} {true_label:<6} {result['label']:<6} {result['confidence']:<10} {mark}")

print(f"\n정확도: {correct}/10 ({correct * 10}%)")



##### Step 5. 에러 테스트 #####

# 784개가 아닌 100개만 전송
response = requests.post(
    "http://localhost:8000/predict",
    json={"pixel_values": [0.0] * 100}
)
print(f"상태 코드: {response.status_code}")  # 422
print(f"에러 메시지: {response.json()['detail'][0]['msg']}")


# 숫자가 아닌 문자열 전달
response = requests.post(
    "http://localhost:8000/predict",
    json={"pixel_values": "이것은 이미지가 아닙니다"}
)
print(f"상태 코드: {response.status_code}")  # 422


# pixel_values 없이 요청
response = requests.post(
    "http://localhost:8000/predict",
    json={"return_probabilities": True}
)
print(f"상태 코드: {response.status_code}")  # 422
print(f"에러: {response.json()['detail'][0]['msg']}")


response = requests.post(
    "http://localhost:8000/predict",
    json={}
)
print(f"상태 코드: {response.status_code}")  # 422



