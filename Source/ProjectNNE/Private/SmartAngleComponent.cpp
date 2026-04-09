// Fill out your copyright notice in the Description page of Project Settings.


#include "SmartAngleComponent.h"

// Sets default values for this component's properties
USmartAngleComponent::USmartAngleComponent()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = false;

	// ...
}


// Called when the game starts
void USmartAngleComponent::BeginPlay()
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

bool USmartAngleComponent::InitializeNNE()
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
    ModelInstance->SetInputTensorShapes({ UE::NNE::FTensorShape::Make({1, 3}) });

    return true;
}

/*
==================================================
📌 [언리얼 C++ 적용을 위한 정규화 값]
입력(X) 평균 (Mean): [1250.4278102988746, 174.9727100042194, 0.509142053445851]
입력(X) 표준편차 (Scale): [851.2049475408038, 165.3701067116099, 0.4999164158724868]
출력(y) 평균 (Mean): [29.59322940646976]
출력(y) 표준편차 (Scale): [15.904859565350147]
==================================================
*/
float USmartAngleComponent::GetSmartAngle(float Distance2D, float HeightDiff)
{
	if (!ModelInstance.IsValid()) return -1.0f;

	float ArcMode = 0.0f;

	// 1. 입력 데이터 정규화 (파이썬 학습 시 사용된 StandardScaler 값 적용)
	// 공식: (입력값 - 평균) / 표준편차
	float ScaledDist = (Distance2D - 1250.4278102988746f) / 851.2049475408038f;
	float ScaledHeight = (HeightDiff - 174.9727100042194f) / 165.3701067116099f;
	float ScaledArcMode = (ArcMode - 0.509142053445851f) / 0.4999164158724868f;

	// 2. NNE 입력/출력 바인딩 설정
	TArray<float> InputData = { ScaledDist, ScaledHeight, ScaledArcMode };
	TArray<float> OutputData = { 0.0f };

	UE::NNE::FTensorBindingCPU InputBinding{ InputData.GetData(), (uint64)InputData.Num() * sizeof(float) };
	UE::NNE::FTensorBindingCPU OutputBinding{ OutputData.GetData(), (uint64)OutputData.Num() * sizeof(float) };

	// 3. AI 추론 실행
	auto Status = ModelInstance->RunSync({ InputBinding }, { OutputBinding });

	if ((int32)Status != 0)
	{
		UE_LOG(LogTemp, Error, TEXT("NNE: Inference Failed!"));
		return 0.0f;
	}

	// 4. 출력 데이터 역정규화 (AI가 뱉은 값을 실제 각도로 변환)
	// 공식: (결과값 * 표준편차) + 평균
	float PredictedAngle = (OutputData[0] * 15.904859565350147f) + 29.59322940646976;

	return PredictedAngle;
}

