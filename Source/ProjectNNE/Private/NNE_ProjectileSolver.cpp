#include "NNE_ProjectileSolver.h"

ANNE_ProjectileSolver::ANNE_ProjectileSolver()
{
	PrimaryActorTick.bCanEverTick = false;
}

void ANNE_ProjectileSolver::BeginPlay()
{
	Super::BeginPlay();
	if (!ModelData) return;

	TWeakInterfacePtr<INNERuntimeCPU> Runtime = UE::NNE::GetRuntime<INNERuntimeCPU>(FString("NNERuntimeORTCpu"));
	if (Runtime.IsValid())
	{
		TSharedPtr<UE::NNE::IModelCPU> Model = Runtime->CreateModelCPU(ModelData);
		if (Model.IsValid())
		{
			ModelInstance = Model->CreateModelInstanceCPU();
		}
	}
}

float ANNE_ProjectileSolver::FireProjectileWithAI(FVector TargetLocation)
{
	// 1. AI 계산에 필요한 입력값 준비
	FVector StartLocation = GetActorLocation(); // 발사 위치
	FVector Diff = TargetLocation - StartLocation;

	float HorizontalDist = FVector2D(Diff.X, Diff.Y).Size();
	float HeightDiff = Diff.Z;

	// 2. AI 솔버로부터 예측 파라미터 가져오기
	float fAngle = GetAILaunchParameters(HorizontalDist, HeightDiff);

	// FVector ForwardDir = Diff.GetSafeNormal();
	// FVector RotationAxis = FVector::CrossProduct(ForwardDir, FVector::UpVector).GetSafeNormal();

	// 3. AI가 준 각도(fAngle)만큼 축을 기준으로 ForwardDir을 회전시킵니다.
	// FVector FinalLaunchDir = ForwardDir.RotateAngleAxis(fAngle, RotationAxis);

	// 최종적으로 각도가 적용된 방향 벡터를 반환합니다.
	return fAngle;
}

float ANNE_ProjectileSolver::GetAILaunchParameters(float HorizontalDist, float HeightDiff)
{
	if (!ModelInstance.IsValid()) return 0.f;

	// --- [중요 1] 입력(X) 정규화 값 (순서: 수평거리, 높이차) ---
	// 파이썬 학습 결과(Loss 0.003 버전)에서 나온 2개의 숫자입니다.
	float X_Means[] = { 2417.5453, -47.6161 };
	float X_Scales[] = { 976.734, 269.7443 };

	TArray<float> InputData;
	// 이제 입력은 수평거리와 높이차 2개만 사용합니다.
	float RawX[] = { HorizontalDist, HeightDiff };

	for (int i = 0; i < 2; ++i) { // 5에서 2로 변경
		float Normalized = (RawX[i] - X_Means[i]);
		if (X_Scales[i] > 0.0001f) Normalized /= X_Scales[i];
		InputData.Add(Normalized);
	}

	// --- [중요 2] 입력 텐서 모양을 {1, 2}로 변경 ---
	// AI 모델이 2개의 특징점(Feature)을 받도록 입구를 좁혔습니다.
	TArray<UE::NNE::FTensorShape> InputShapes;
	InputShapes.Add(UE::NNE::FTensorShape::Make({ 1, 2 }));

	if (ModelInstance->SetInputTensorShapes(InputShapes) != UE::NNE::IModelInstanceCPU::ESetInputTensorShapesStatus::Ok)
	{
		UE_LOG(LogTemp, Error, TEXT("입력 모양 설정 실패! (2개여야 함)"));
		return 0.f;
	}

	// 3. 실행 로직 (동일)
	TArray<float> OutputData;
	OutputData.SetNumZeroed(2);

	UE::NNE::FTensorBindingCPU InputBinding;
	InputBinding.Data = InputData.GetData();
	InputBinding.SizeInBytes = (uint64)InputData.Num() * sizeof(float);

	UE::NNE::FTensorBindingCPU OutputBinding;
	OutputBinding.Data = OutputData.GetData();
	OutputBinding.SizeInBytes = (uint64)OutputData.Num() * sizeof(float);

	UE::NNE::IModelInstanceCPU::ERunSyncStatus Status = ModelInstance->RunSync({ InputBinding }, { OutputBinding });

	if (Status != UE::NNE::IModelInstanceCPU::ERunSyncStatus::Ok)
	{
		UE_LOG(LogTemp, Error, TEXT("AI 추론 실행 실패!"));
		return 0.f;
	}

	// --- [중요 3] 출력(y) 역정규화 값 (순서: 각도, 힘) ---
	// 파이썬 학습 결과(Loss 0.003 버전)의 출력 정규화 수치입니다.
	float Y_Means[] = { 11.4014 };
	float Y_Scales[] = { 5.1054 };

	float fAngle = (OutputData[0] * Y_Scales[0]) + Y_Means[0];

	UE_LOG(LogTemp, Warning, TEXT("AI 결과 -> Angle: %f"), fAngle);

	return fAngle;
}