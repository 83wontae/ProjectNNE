# 비선형성(Non-linearity) 모델
# 인공지능이 단순히 "몇 배를 곱하는 계산기"를 넘어 복잡한 판단을 내리게 하려면 '직선을 구부려주는' 장치. 가장 대표적인 것이 ReLU(Rectified Linear Unit).
# ReLU는 입력이 0보다 작으면 0, 0 이상이면 그대로 출력하는 함수입니다. 이렇게 하면 모델이 단순한 선형 관계를 넘어 다양한 패턴을 학습할 수 있습니다.

import torch
import torch.nn as nn
import torch.optim as optim

class ThresholdModel(nn.Module):
    def __init__(self):
        super(ThresholdModel, self).__init__()
        # 입력 1 -> 출력 1
        self.linear = nn.Linear(1, 1)
        # 활성화 함수 추가! (직선을 구부립니다)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x

# 1. 데이터 준비 (공격력이 10 이하면 0, 그 이상이면 증가)
x_train = torch.tensor([[5.0], [8.0], [10.0], [15.0], [20.0], [30.0]], dtype=torch.float32)
y_train = torch.tensor([[0.0], [0.0], [0.0],  [5.0],  [10.0], [20.0]], dtype=torch.float32)

model = ThresholdModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 2. 학습
for epoch in range(5000):
    pred = model(x_train)
    loss = criterion(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 3. 테스트
model.eval()
print(f"공격력 7 예측: {model(torch.tensor([[7.0]])).item():.2f}") # 0에 가까워야 함
print(f"공격력 25 예측: {model(torch.tensor([[25.0]])).item():.2f}") # 15에 가까워야 함

# 4. ONNX 내보내기
torch.onnx.export(model, torch.randn(1, 1), "NonLinearModel.onnx")