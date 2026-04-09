// Fill out your copyright notice in the Description page of Project Settings.


#include "NNE_SimpleDouble.h"

// Sets default values
ANNE_SimpleDouble::ANNE_SimpleDouble()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

// Called when the game starts or when spawned
void ANNE_SimpleDouble::BeginPlay()
{
    if (InitializeNNE())
    {
        UE_LOG(LogTemp, Log, TEXT("NNE: 모델 초기화 성공!"));
        OnModelReady();
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("NNE: 모델 초기화 실패!"));
    }

	Super::BeginPlay();
}

float ANNE_SimpleDouble::PredictDouble(float InputValue)
{
    if (!ModelInstance.IsValid()) return -1.0f;

    // 1. 데이터 준비 (파이썬 dummy_input shape [1, 1]에 맞춤)
    TArray<float> InputData = { InputValue };
    TArray<float> OutputData = { 0.0f };

    UE::NNE::FTensorBindingCPU InputBinding{ InputData.GetData(), (uint64)InputData.Num() * sizeof(float) };
    UE::NNE::FTensorBindingCPU OutputBinding{ OutputData.GetData(), (uint64)OutputData.Num() * sizeof(float) };

    // 3. 동기식 추론 실행 (RunSync)
    if ((int32)ModelInstance->RunSync({ InputBinding }, { OutputBinding }) != 0)
    {
        UE_LOG(LogTemp, Error, TEXT("NNE: 추론 실행 중 오류 발생"));
        return -1.0f;
    }

    // 결과 반환 (2.0이 곱해진 값)
    return OutputData[0];
}

bool ANNE_SimpleDouble::InitializeNNE()
{
    if (!ModelData) return false;

    // 1. CPU 런타임 인터페이스 가져오기 (NNERuntimeORTCpu 사용)
    TWeakInterfacePtr<INNERuntimeCPU> RuntimePtr = UE::NNE::GetRuntime<INNERuntimeCPU>(TEXT("NNERuntimeORTCpu"));
    if (!RuntimePtr.IsValid()) return false;

    // 2. 모델 생성
    TSharedPtr<UE::NNE::IModelCPU> Model = RuntimePtr->CreateModelCPU(ModelData);
    if (!Model.IsValid()) return false;

    // 3. 실행 가능한 인스턴스 생성
    ModelInstance = Model->CreateModelInstanceCPU();
    if (!ModelInstance.IsValid()) return false;

    // (1, 1) 텐서 모양 설정
    ModelInstance->SetInputTensorShapes({ UE::NNE::FTensorShape::Make({1, 1}) });

    return true;
}
