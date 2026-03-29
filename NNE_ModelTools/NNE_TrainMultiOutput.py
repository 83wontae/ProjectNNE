# 입력은 하나지만 결과는 여러 개인 경우가 매우 흔하며, 이를 "일대다(One-to-Many)" 또는 "다중 출력(Multi-Output)" 모델

import torch
import torch.nn as nn
import torch.optim as optim

# 1. 데이터 준비 (입력: [공격력], 출력: [데미지, 크리티컬확률, 흔들림강도])
# 규칙 1 (데미지): 공격력 * 1.5 + 5
# 규칙 2 (크리티컬): 공격력 * 0.01 (최대 1.0)
# 규칙 3 (흔들림): 공격력 * 0.1
x_train = torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0]], dtype=torch.float32)

y_train = torch.tensor([
    [20.0, 0.1, 1.0], 
    [35.0, 0.2, 2.0], 
    [50.0, 0.3, 3.0], 
    [65.0, 0.4, 4.0], 
    [80.0, 0.5, 5.0]
], dtype=torch.float32)

# 2. 모델 정의 (입력 1개 -> 출력 3개)
# Linear(1, 3)은 내부적으로 3개의 서로 다른 가중치(w)와 편향(b)을 가집니다.
model = nn.Linear(in_features=1, out_features=3, bias=True)

# 3. 학습 설정
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. 학습 시작
print("다중 출력 모델 학습 시작...")
for epoch in range(3000):
    prediction = model(x_train)
    loss = criterion(prediction, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch+1}/3000], Loss: {loss.item():.6f}")

# 5. 모델 검증
model.eval()
with torch.no_grad():
    test_val = 25.0
    test_input = torch.tensor([[test_val]])
    pred = model(test_input)
    
    print(f"\n--- 테스트 결과 (입력 공격력: {test_val}) ---")
    print(f"예측 데미지: {pred[0][0]:.2f} (기대값: 42.5)")
    print(f"예측 크리티컬: {pred[0][1]:.2f} (기대값: 0.25)")
    print(f"예측 흔들림: {pred[0][2]:.2f} (기대값: 2.5)")

# 6. ONNX 내보내기
dummy_input = torch.randn(1, 1)
output_file = "MultiOutputModel.onnx"

torch.onnx.export(
    model, 
    dummy_input, 
    output_file,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input_0'],
    output_names=['output_0']
)

print(f"\n파일 저장 완료: {output_file}")