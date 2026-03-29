import torch
import torch.nn as nn
import torch.optim as optim

# 1. 데이터 준비 (입력: [공격력, 숙련도], 출력: [최종 데미지])
# 규칙: 데미지 = (공격력 * 1.5) + (숙련도 * 2.0) + 5 (기본값)
x_train = torch.tensor([
    [10.0, 5.0], [20.0, 10.0], [30.0, 15.0], [40.0, 20.0], [50.0, 25.0]
])
y_train = torch.tensor([
    [30.0], [65.0], [100.0], [135.0], [170.0]
])

# 2. 모델 정의 (입력 2개, 출력 1개, Bias 사용)
model = nn.Linear(2, 1, bias=True)

# 3. 학습 설정
# SGD (Stochastic Gradient Descent, 확률적 경사 하강법)
# Adam (Adaptive Moment Estimation)
criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.1) # 학습률을 높여도 Adam은 안전합니다.

# 4. 학습 시작 (2000번 반복)
for epoch in range(2000):
    prediction = model(x_train)
    loss = criterion(prediction, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 결과 확인
w = model.weight.data
b = model.bias.data
print(f"학습 완료! 가중치: {w}, 편향: {b}")

# 5. ONNX로 내보내기
dummy_input = torch.randn(1, 2)
torch.onnx.export(model, dummy_input, "MultiDamage.onnx")
print("MultiDamage.onnx 저장 완료!")