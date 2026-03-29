import torch
import torch.nn as nn
import torch.onnx
import os

# 1. 간단한 신경망 모델 정의
class SimpleDoubleModel(nn.Module):
    def __init__(self):
        super(SimpleDoubleModel, self).__init__()
        # 입력값에 가중치를 곱하는 선형 레이어 (Bias는 계산의 단순함을 위해 제거)
        self.linear = nn.Linear(1, 1, bias=False)
        # 가중치를 2.0으로 강제 설정 (입력 x 2 = 출력)
        self.linear.weight.data.fill_(2.0)

    def forward(self, x):
        return self.linear(x)

# 2. 모델 인스턴스 생성 및 평가 모드 전환
model = SimpleDoubleModel()
model.eval()

# 3. 모델 구조를 파악하기 위한 가짜 입력 데이터 (Shape: 1x1)
# NNE는 이 데이터의 형태를 보고 입력 텐서의 크기를 결정합니다.
dummy_input = torch.randn(1, 1)

# 4. ONNX 파일로 내보내기
output_filename = "SimpleDouble.onnx"

torch.onnx.export(
    model, 
    dummy_input, 
    output_filename,
    export_params=True,        # 학습된 가중치를 함께 저장
    opset_version=14,          # 언리얼 NNE(ORT)와 호환성이 좋은 버전
    do_constant_folding=True,  # 모델 최적화
    input_names=['input_0'],   # ★ 중요: 언리얼에서 부를 입력 노드 이름
    output_names=['output_0'], # ★ 중요: 언리얼에서 부를 출력 노드 이름
    dynamic_axes={'input_0' : {0 : 'batch_size'}, 'output_0' : {0 : 'batch_size'}} # 배치 사이즈 유연성
)

print(f"성공: {os.path.abspath(output_filename)} 파일이 생성되었습니다.")
