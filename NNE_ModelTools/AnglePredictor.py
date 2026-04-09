import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd # 데이터를 표(Table) 형태로 다루는 도구
import numpy as np # 고성능 공학용 계산기
import os
from sklearn.preprocessing import StandardScaler # 정규화 도구

# ==========================================
# 1. 데이터 로드 및 전처리
# ==========================================
csv_name = "ProjectileTrainingData.csv"
base_dir = os.path.dirname(os.path.abspath(__file__))

# 파일 경로 찾기 (현재 폴더 혹은 언리얼 Saved 폴더)
csv_path = os.path.join(base_dir, "..", "Saved", csv_name) if not os.path.exists(csv_name) else csv_name

if not os.path.exists(csv_path):
    print(f"❌ 에러: {csv_path} 파일을 찾을 수 없습니다! 언리얼에서 데이터를 먼저 수집하세요.")
    exit()

# 데이터 읽기 (header=None: 첫 줄부터 데이터인 경우)
data = pd.read_csv(csv_path, header=None)
print(f"📊 수집된 총 데이터: {len(data)}개")

hit_data = data.dropna()

if len(hit_data) < 10:
    print("⚠️ 데이터가 부족합니다! 언리얼에서 더 많은 '성공' 데이터를 모아주세요.")
    exit()

# [0]거리, [1]높이차, [2]각도, [3]고각 여부
# 입력(X): 거리, 높이차 / 출력(y): 각도
X = data[[0, 1, 3]].values
y = data[2].values.reshape(-1, 1)

# 데이터 정규화 (Standardization)
# 인공지능이 큰 숫자에 당황하지 않게 0 주변으로 압축해주는 과정입니다.
scaler_X = StandardScaler() # 정규화 도구 생성
scaler_y = StandardScaler() # 정규화 도구 생성

X_scaled = scaler_X.fit_transform(X) # 평균이 0, 표준편차가 1인 상태로 변환
y_scaled = scaler_y.fit_transform(y) # 평균이 0, 표준편차가 1인 상태로 변환

# [매우 중요] 이 값들을 메모해두었다가 언리얼 C++ 코드에 입력해야 합니다!
print("\n" + "="*50)
print("📌 [언리얼 C++ 적용을 위한 정규화 값]")
print(f"입력(X) 평균 (Mean): {scaler_X.mean_.tolist()}")
print(f"입력(X) 표준편차 (Scale): {scaler_X.scale_.tolist()}")
print(f"출력(y) 평균 (Mean): {scaler_y.mean_.tolist()}")
print(f"출력(y) 표준편차 (Scale): {scaler_y.scale_.tolist()}")
print("="*50 + "\n")

X_tensor = torch.tensor(X_scaled, dtype=torch.float32) # 입력 데이터 준비
y_tensor = torch.tensor(y_scaled, dtype=torch.float32) # 출력 데이터 준비

# ==========================================
# 2. 신경망 모델 정의 (Regression)
# ==========================================
class AnglePredictor(nn.Module):
    def __init__(self):
        super(AnglePredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),   # 입력층 : 입력 3개 (거리, 높이차, 고각 여부)
            nn.ReLU(),
            nn.Linear(64, 64),  # 은닉층 : 64개 뉴런
            nn.ReLU(),
            nn.Linear(64, 32),  # 은닉층 : 32개 뉴런
            nn.ReLU(),
            nn.Linear(32, 1)    # 출력층 : 출력 1개 (예측 각도)
        )

    def forward(self, x):
        return self.net(x)

model = AnglePredictor()

# ==========================================
# 3. 학습 시작
# ==========================================
criterion = nn.MSELoss() # 정답과의 수치 차이를 계산하는 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("🚀 학습 시작...")
model.train()
for epoch in range(20000):
    prediction = model(X_tensor)
    loss = criterion(prediction, y_tensor)

    # X_tensor의 3번째 열이 0(저각)인 데이터에 가중치 2배 부여  [:, 2] = [1,2,3,4,5,6,7,8,9,10] , [:, 2:3] = [[0],[0],[1],[1],[0],[0],[1],[1],[0],[0]]
    weights = torch.where(X_tensor[:, 2:3] == 0, 2.0, 1.0) # 2 <= x < 3
    weighted_loss = (loss * weights).mean()

    optimizer.zero_grad()
    weighted_loss.backward()
    optimizer.step()

    if (epoch + 1) % 2000 == 0:
        print(f"Epoch [{epoch+1}/20000], Loss: {loss.item():.6f}")

print("✨ 학습 완료!")

# ==========================================
# 4. 학습된 모델을 ONNX 파일로 저장 (내보내기)
# ==========================================
model.eval()

with torch.no_grad(): # 추론 시 불필요한 메모리 사용 방지
    dummy_input = torch.randn(1, 3) # 모델 구조를 파악하기 위한 가짜 데이터
    output_filename = "AnglePredictor.onnx"
    torch.onnx.export(model, dummy_input, output_filename)

print(f"학습된 모델이 {os.path.abspath(output_filename)} 파일로 저장되었습니다!")