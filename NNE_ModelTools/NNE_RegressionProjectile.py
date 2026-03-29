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
    csv_name = os.path.join(base_dir, "..", "Saved", "ProjectileTrainingData.csv")

if not os.path.exists(csv_name):
    print(f"❌ 에러: {csv_name} 파일을 찾을 수 없습니다!")
    exit()

# 2. 데이터 로드
data = pd.read_csv(csv_name, header=None)
print(f"📊 총 데이터 개수: {len(data)}개")

# 3. 데이터 전처리 (성공 데이터 추출)
# 8번째 열(인덱스 7)이 Hit 여부입니다.
hit_column_idx = 7 
data[hit_column_idx] = pd.to_numeric(data[hit_column_idx], errors='coerce')
hit_data = data[data[hit_column_idx] == 1].dropna()
print(f"✅ 찾아낸 성공(Hit) 데이터: {len(hit_data)}개")

if len(hit_data) < 10:
    print("❌ 데이터가 너무 적습니다. 더 수집해주세요.")
    exit()

# 4. 입력(X)과 출력(y) 분리
# 입력(X): 수평거리(0), 높이차(1), 무게(4), 반지름(5), 타겟반지름(6) -> 총 5개
X = hit_data.iloc[:, [0, 1, 4, 5, 6]].values
# 출력(y): 각도(2), 힘(3) -> 총 2개
y = hit_data.iloc[:, [2, 3]].values

# 5. 정규화 (StandardScaler)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# [언리얼용] 나중에 C++ 코드에 복사할 값들 출력
print("\n--- [언리얼용] 입력(X) 정규화 값 (5개 항목) ---")
print("Means (평균):", scaler_X.mean_.tolist())
print("Scales (표준편차):", scaler_X.scale_.tolist())
print("\n--- [언리얼용] 출력(y) 역정규화 값 (2개 항목) ---")
print("Means (평균):", scaler_y.mean_.tolist())
print("Scales (표준편차):", scaler_y.scale_.tolist())

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# 6. 모델 정의
# [중요] 입력 특징이 5개이므로 첫 번째 Linear의 입력을 5로 설정합니다.
model = nn.Sequential(
    nn.Linear(5, 256), 
    nn.LeakyReLU(0.1),
    nn.Linear(256, 128),
    nn.LeakyReLU(0.1),
    nn.Linear(128, 64),
    nn.LeakyReLU(0.1),
    nn.Linear(64, 2)
)

# 7. 학습 설정
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)

# 8. 학습 루프
print("\n🚀 학습 시작...")
for epoch in range(10000):
    pred = model(X_tensor)
    loss = criterion(pred, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f"에포크 [{epoch+1}/10000], 오차(Loss): {loss.item():.6f}")

# 9. ONNX 모델로 내보내기
model.eval()
dummy_input = torch.randn(1, 5) # 입력 크기 5로 변경
onnx_path = "ProjectileSolver_Final.onnx"
torch.onnx.export(model, dummy_input, onnx_path, verbose=False,
                  input_names=['input'], output_names=['output'])

print(f"\n✅ 학습 완료! 모델 저장됨: {onnx_path}")