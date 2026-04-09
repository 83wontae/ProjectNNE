import torch
import torch.nn as nn
import torch.onnx
import os

# 1. 간단한 신경망 모델 정의
# nn.Module을 상속 받아서 SimpleDoubleModel을 정의한다.
class SimpleDoubleModel(nn.Module):
    def __init__(self): # 생성자(태어났을 때, 자판기에 전원 On)
        super(SimpleDoubleModel, self).__init__() # super(부모 모델 실행)
        # 입력값에 가중치를 곱하는 선형 레이어 (Bias는 계산의 단순함을 위해 제거)
        self.linear = nn.Linear(1, 1, bias=False)
        # 가중치를 2.0으로 강제 설정 (입력 x 2 = 출력)
        self.linear.weight.data.fill_(2.0)

    def forward(self, x): # 모듈에 값이 들어왔을때(모듈(함수)의 실행, 자판기에 동전 투입)
        return self.linear(x) # self.linear를 실행해서 반환

# 2. 모델 인스턴스 생성 및 평가 모드 전환
model = SimpleDoubleModel() # 객체 생성
model.eval() # 평가/추론 모드

# 3. 트레이싱(Tracing) - 모델 구조를 파악하기 위한 가짜 입력 데이터 (Shape: 1x1)
# NNE는 이 데이터의 형태를 보고 입력 텐서의 크기를 결정합니다.
dummy_input = torch.randn(1, 1)

# 4. ONNX 파일로 내보내기
output_filename = "SimpleDouble.onnx"
torch.onnx.export(model, dummy_input, output_filename)

print(f"성공: {os.path.abspath(output_filename)} 파일이 생성되었습니다.")