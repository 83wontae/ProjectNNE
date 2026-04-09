import torch
import torch.nn as nn
import torch.optim as optim
import os

# 1. 데이터 준비 (입력 x와 정답 y)
# 1을 넣으면 2, 2를 넣으면 4, 3을 넣으면 6이 나온다는 '힌트'만 줍니다.
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 2. 모델 정의 (가중치를 초기화하지 않음. 처음엔 랜덤값이 들어감)
model = nn.Linear(1, 1, bias=False) # y = w(가중치)x + b(편향)

# 3. 손실 함수와 최적화 도구 (정답과 얼마나 차이나는지 계산하고 수정하는 도구)
criterion = nn.MSELoss() # 선생님의 빨간 펜 (오차 계산)
optimizer = optim.SGD(model.parameters(), lr=0.01) # 학습 매니저 (성적 향상 도우미)

# 4. 학습 시작 (1000번 반복해서 정답을 찾아라!)
model.train() # 학습 모드
for epoch in range(1000):
    prediction = model(x_train) # 문제 풀기
    loss = criterion(prediction, y_train) # 성적 발표

    optimizer.zero_grad() # 칠판 지우기
    loss.backward()  # 학습
    optimizer.step() # 두뇌 업데이트

print(f"학습 완료! 찾아낸 가중치: {model.weight.item():.4f}")
model.eval() # 평가 모드

# 5. 학습된 모델을 ONNX 파일로 저장 (내보내기)
with torch.no_grad(): # 추론 시 불필요한 메모리 사용 방지
    dummy_input = torch.randn(1, 1) # 모델 구조를 파악하기 위한 가짜 데이터
    output_filename = "SimpleDoubleTrained.onnx"
    torch.onnx.export(model, dummy_input, output_filename)

print(f"학습된 모델이 {os.path.abspath(output_filename)} 파일로 저장되었습니다!")