import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# 1. 파일 경로 설정
csv_name = "ProjectileTrainingData.csv"
if not os.path.exists(csv_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_name = os.path.join(base_dir, "..", "Saved", csv_name)

if not os.path.exists(csv_name):
    print(f"❌ 에러: {csv_name} 파일을 찾을 수 없습니다!")
    exit()

# 2. 데이터 로드
# 데이터 구조: [0]수평거리, [1]높이차, [2]각도, [3]성공여부 (총 4개 열)
data = pd.read_csv(csv_name, header=None)
print(f"📊 수집된 총 데이터: {len(data)}개")

# 3. 데이터 전처리 (성공 데이터만 추출)
# 이제 Hit 여부는 4번째 열(인덱스 3)에 있습니다.
hit_column_idx = 3 
data[hit_column_idx] = pd.to_numeric(data[hit_column_idx], errors='coerce')
hit_data = data[data[hit_column_idx] == 1].dropna()
print(f"✅ 학습에 사용할 성공 데이터: {len(hit_data)}개")

if len(hit_data) < 5:
    print("❌ 성공 데이터가 부족합니다. 언리얼에서 힘을 고정한 채로 데이터를 더 수집해주세요.")
    exit()

# 4. 입력(X)과 출력(y) 분리
# 입력(X): [0]수평거리, [1]높이차 -> 2개
X = hit_data.iloc[:, [0, 1]].values
# 출력(y): [2]각도 -> 1개 (힘은 고정값이므로 예측에서 제외)
y = hit_data.iloc[:, [2]].values

# 5. 정규화 (StandardScaler)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# [언리얼 C++ 복사용] 정규화 파라미터 출력
print("\n" + "="*50)
print("📌 [언리얼 C++용] 입력(X) 정규화 값 (순서: 수평거리, 높이차)")
print("Means (평균):", [round(x, 4) for x in scaler_X.mean_.tolist()])
print("Scales (표준편차):", [round(x, 4) for x in scaler_X.scale_.tolist()])
print("-" * 50)
print("📌 [언리얼 C++용] 출력(y) 역정규화 값 (항목: 각도)")
print("Mean (평균):", [round(y[0], 4) for y in scaler_y.mean_.reshape(-1, 1).tolist()])
print("Scale (표준편차):", [round(y[0], 4) for y in scaler_y.scale_.reshape(-1, 1).tolist()])
print("="*50 + "\n")

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# 6. 신경망 모델 정의
# 입력 2개 -> 출력 1개 (각도)
model = nn.Sequential(
    nn.Linear(2, 256), 
    nn.LeakyReLU(0.1),
    nn.Linear(256, 128),
    nn.LeakyReLU(0.1),
    nn.Linear(128, 64),
    nn.LeakyReLU(0.1),
    nn.Linear(64, 1) # 마지막 노드를 1로 변경
)

# 7. 학습 설정
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)

# 8. 학습 진행
print("🚀 각도 예측 AI 학습 중...")
for epoch in range(10000):
    pred = model(X_tensor)
    loss = criterion(pred, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f"에포크 [{epoch+1}/10000], 오차(Loss): {loss.item():.6f}")

# 9. ONNX 모델 내보내기
model.eval()
dummy_input = torch.randn(1, 2) 
onnx_path = "ProjectileSolver_AngleOnly.onnx"
torch.onnx.export(model, dummy_input, onnx_path, verbose=False,
                  input_names=['input'], output_names=['angle_output'])

print(f"\n✅ 학습 완료! 각도 전용 모델 저장됨: {onnx_path}")