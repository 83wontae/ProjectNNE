# 명중 판독기
# 언리얼에서 발사체의 명중 여부를 예측하는 AI 모델을 학습시키는 스크립트입니다.
# ex) 플레이어가 조준하고 있을 때 **"명중 확률 80%"**라고 화면에 띄워주기.
# ex) 가이드 라인 색깔을 **초록색(높음)**이나 **빨간색(낮음)**으로 바꿔주기.

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler

# 1. 파일 경로 확인 (언리얼 Saved 폴더 경로에 맞춰 수정하세요)
# 예: "C:/UnrealProject/ProjectNNE/Saved/ProjectileTrainingData.csv"
# 1. 이 파일(.py)의 절대 경로를 계산
base_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 그 경로 뒤에 파일 이름을 붙임
csv_path = os.path.join(base_dir, "..", "Saved", "ProjectileTrainingData.csv")

if not os.path.exists(csv_path):
    print(f"오류: {csv_path} 파일을 찾을 수 없습니다. 언리얼에서 먼저 데이터를 생성해주세요!")
    exit()

data = pd.read_csv(csv_path, header=None)

# 입력(X): 0~5번 열 (Distance, Angle, Impulse, Weight, Radius, TargetRadius)
# 출력(y): 6번 열 (Hit/Miss)
X = data.iloc[:, 0:6].values
y = data.iloc[:, 6].values.reshape(-1, 1)

# 2. 데이터 정규화 (Normalization)
# 큰 숫자(Impulse 등)를 작은 범위로 압축하여 학습 효율을 높입니다.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# [중요] 나중에 언리얼 C++에서 사용할 평균과 표준편차입니다. 꼭 기록해두세요!
print("\n--- 언리얼 C++ 적용을 위한 정규화 값 ---")
print(f"평균 (Mean): {scaler.mean_.tolist()}")
print(f"표준편차 (Scale): {scaler.scale_.tolist()}")
print("--------------------------------------\n")

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 3. 딥러닝 모델 정의 (은닉층 2개)
class ProjectilePredictor(nn.Module):
    def __init__(self):
        super(ProjectilePredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() # 0(0%) ~ 1(100%) 사이 확률값 출력
        )

    def forward(self, x):
        return self.net(x)

# 4. 학습 설정
model = ProjectilePredictor()
criterion = nn.BCELoss() # 이진 분류용 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. 학습 시작
print("AI 학습을 시작합니다...")
for epoch in range(5000):
    prediction = model(X_tensor)
    loss = criterion(prediction, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f"에포크 [{epoch+1}/5000], 손실률(Loss): {loss.item():.6f}")

# 6. ONNX로 저장
model.eval()
dummy_input = torch.randn(1, 6)
torch.onnx.export(model, dummy_input, "ProjectileHit.onnx")
print("\n학습 완료! 'ProjectileHit.onnx' 파일이 생성되었습니다.")