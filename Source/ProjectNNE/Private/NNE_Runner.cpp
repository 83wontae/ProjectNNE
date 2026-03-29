#include "NNE_Runner.h"

ANNE_Runner::ANNE_Runner() : BatchSize(1), FeatureCount(2)
{
    PrimaryActorTick.bCanEverTick = false;
}

void ANNE_Runner::BeginPlay()
{
    // 순차적 체크 방식을 통한 초기화 실행
    if (InitializeNNE())
    {
        OnModelInitialized();
        UE_LOG(LogTemp, Warning, TEXT("NNE: Initialization SUCCESS and Event Fired!"));
    }

    Super::BeginPlay();
}

bool ANNE_Runner::InitializeNNE()
{
    if (!ModelData)
    {
        UE_LOG(LogTemp, Error, TEXT("NNE: ModelData is null!"));
        return false;
    }

    if (!CreateModelInstance())
    {
        UE_LOG(LogTemp, Error, TEXT("NNE: Failed to create Model Instance!"));
        return false;
    }

    // 변수화된 BatchSize와 FeatureCount를 사용하여 텐서 모양 설정
    TArray<uint32> Shape = { (uint32)BatchSize, (uint32)FeatureCount };
    ModelInstance->SetInputTensorShapes({ UE::NNE::FTensorShape::Make(Shape) });

    return true;
}

bool ANNE_Runner::CreateModelInstance()
{
    TWeakInterfacePtr<INNERuntimeCPU> RuntimePtr = UE::NNE::GetRuntime<INNERuntimeCPU>(TEXT("NNERuntimeORTCpu"));
    if (!RuntimePtr.IsValid()) return false;

    TSharedPtr<UE::NNE::IModelCPU> Model = RuntimePtr->CreateModelCPU(ModelData);
    if (!Model.IsValid()) return false;

    ModelInstance = Model->CreateModelInstanceCPU();
    return ModelInstance.IsValid();
}

float ANNE_Runner::PredictDouble(float InputValue)
{
    if (!ModelInstance.IsValid()) return -1.0f;

    TArray<float> InputData = { InputValue };
    TArray<float> OutputData = { 0.0f };

    UE::NNE::FTensorBindingCPU InputBinding{ InputData.GetData(), InputData.Num() * sizeof(float) };
    UE::NNE::FTensorBindingCPU OutputBinding{ OutputData.GetData(), OutputData.Num() * sizeof(float) };

    auto Status = ModelInstance->RunSync({ InputBinding }, { OutputBinding });
    return ((int32)Status == 0) ? OutputData[0] : -1.0f;
}

float ANNE_Runner::PredictDamage(float Strength, float Skill)
{
    if (!ModelInstance.IsValid()) return -1.0f;

    TArray<float> InputData = { Strength, Skill };
    TArray<float> OutputData = { 0.0f };

    // 바인딩 시 데이터 개수가 FeatureCount와 일치해야 함에 주의하세요.
    UE::NNE::FTensorBindingCPU InputBinding{ InputData.GetData(), InputData.Num() * sizeof(float) };
    UE::NNE::FTensorBindingCPU OutputBinding{ OutputData.GetData(), OutputData.Num() * sizeof(float) };

    auto Status = ModelInstance->RunSync({ InputBinding }, { OutputBinding });

    if ((int32)Status != 0)
    {
        UE_LOG(LogTemp, Error, TEXT("NNE: PredictDamage RunSync Failed!"));
        return -1.0f;
    }

    return OutputData[0];
}

TArray<float> ANNE_Runner::PredictMultiple(float InputValue)
{
    TArray<float> InputData = { InputValue };

    // 결과값을 3개 받아야 하므로 크기를 3으로 설정!
    TArray<float> OutputData;
    OutputData.SetNumZeroed(3);

    UE::NNE::FTensorBindingCPU InputBinding{ InputData.GetData(), InputData.Num() * sizeof(float) };
    UE::NNE::FTensorBindingCPU OutputBinding{ OutputData.GetData(), OutputData.Num() * sizeof(float) };

    ModelInstance->RunSync({ InputBinding }, { OutputBinding });

    // 이제 OutputData[0], OutputData[1], OutputData[2]에 각각의 결과가 들어있습니다.
    return OutputData;
}
