import torch
import torch.nn as nn
import torch.optim as optim

# 1. 데이터 준비 (입력 x와 정답 y)
# 1을 넣으면 2, 2를 넣으면 4, 3을 넣으면 6이 나온다는 '힌트'만 줍니다.
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 2. 모델 정의 (가중치를 초기화하지 않음. 처음엔 랜덤값이 들어감)
model = nn.Linear(1, 1, bias=False)

# 3. 손실 함수와 최적화 도구 (정답과 얼마나 차이나는지 계산하고 수정하는 도구)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 학습 시작 (1000번 반복해서 정답을 찾아라!)
for epoch in range(1000):
    prediction = model(x_train)
    loss = criterion(prediction, y_train)
    
    optimizer.zero_grad()
    loss.backward()  # <--- 이게 바로 인공지능이 '배우는' 과정(역전파)입니다.
    optimizer.step()

print(f"학습 완료! 찾아낸 가중치: {model.weight.item():.4f}")

# 5. 학습된 모델을 ONNX 파일로 저장 (내보내기)
dummy_input = torch.randn(1, 1) # 모델 구조를 파악하기 위한 가짜 데이터
torch.onnx.export(model, dummy_input, "SimpleDouble_Trained.onnx")

print("학습된 모델이 'SimpleDouble_Trained.onnx' 파일로 저장되었습니다!")