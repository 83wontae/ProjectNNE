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

FLaunchParams ANNE_ProjectileSolver::GetAILaunchParameters(float HorizontalDist, float HeightDiff, float Weight, float Radius, float TargetRad)
{
	FLaunchParams Results;
	if (!ModelInstance.IsValid()) return Results;

	// ==========================================================
	// [필독!] 파이썬 학습 완료 후 출력된 숫자를 여기에 정확히 복사하세요.
	// 순서: [수평거리, 높이차, 무게, 반지름, 타겟반지름]
	// ==========================================================
	float X_Means[] = { 968.8f, -135.34f, 40.f, 32.f, 32.f }; // <-- 파이썬의 Means (5개)
	float X_Scales[] = { 428.8f, 327.79f, 1.f, 1.f, 1.f };   // <-- 파이썬의 Scales (5개)

	TArray<float> InputData;
	float RawX[] = { HorizontalDist, HeightDiff, Weight, Radius, TargetRad };

	for (int i = 0; i < 5; ++i) {
		float Normalized = (RawX[i] - X_Means[i]);
		// 분모가 0이 되는 것을 방지
		if (X_Scales[i] > 0.0001f) Normalized /= X_Scales[i];
		InputData.Add(Normalized);
	}

	// 입력 모양 설정 (1행 5열)
	TArray<UE::NNE::FTensorShape> InputShapes;
	InputShapes.Add(UE::NNE::FTensorShape::Make({ 1, 5 }));

	if (ModelInstance->SetInputTensorShapes(InputShapes) != UE::NNE::IModelInstanceCPU::ESetInputTensorShapesStatus::Ok)
	{
		UE_LOG(LogTemp, Error, TEXT("입력 모양 설정 실패! (5개여야 함)"));
		return Results;
	}

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
		return Results;
	}

	// ==========================================================
	// [필독!] 파이썬 출력 결과에 나온 Y 역정규화 값으로 교체하세요.
	// 순서: [각도(Angle), 힘(Impulse)]
	// ==========================================================
	float Y_Means[] = { -3.8f, 128158.45f };   // <-- 파이썬 출력값 참고
	float Y_Scales[] = { 26.1f, 41720.9f };  // <-- 파이썬 출력값 참고

	Results.Angle = (OutputData[0] * Y_Scales[0]) + Y_Means[0];
	Results.Impulse = (OutputData[1] * Y_Scales[1]) + Y_Means[1];
	Results.bSuccess = true;

	// 디버깅용 로그: 이제 Impulse가 50000~200000 사이로 나와야 정상입니다.
	UE_LOG(LogTemp, Warning, TEXT("AI 결과 -> Angle: %f, Impulse: %f"), Results.Angle, Results.Impulse);

	return Results;
}