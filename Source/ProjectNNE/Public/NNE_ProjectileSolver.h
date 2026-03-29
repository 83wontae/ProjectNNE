// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
// NNE 필수 헤더
#include "NNE.h"
#include "NNERuntimeCPU.h"
#include "NNEModelData.h"
#include "NNE_ProjectileSolver.generated.h"

/**
 * AI의 예측 결과(각도와 힘)를 담는 구조체입니다.
 */
USTRUCT(BlueprintType)
struct FLaunchParams
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "NNE|Result")
	float Angle = 0.0f; // AI가 계산한 발사 각도

	UPROPERTY(BlueprintReadOnly, Category = "NNE|Result")
	float Impulse = 0.0f; // AI가 계산한 발사 강도

	UPROPERTY(BlueprintReadOnly, Category = "NNE|Result")
	bool bSuccess = false; // 추론 성공 여부
};

UCLASS()
class PROJECTNNE_API ANNE_ProjectileSolver : public AActor
{
	GENERATED_BODY()

public:
	ANNE_ProjectileSolver();

	// 에디터에서 학습된 .onnx 에셋을 여기에 할당합니다.
	UPROPERTY(EditAnywhere, Category = "NNE")
	TObjectPtr<UNNEModelData> ModelData;

	/**
	 * 상황 데이터를 입력하면 최적의 [각도, 힘]을 반환하는 핵심 함수입니다.
	 */
	UFUNCTION(BlueprintCallable, Category = "NNE|Solver")
	FLaunchParams GetAILaunchParameters(float HorizontalDist, float HeightDiff, float Weight, float Radius, float TargetRad);

protected:
	virtual void BeginPlay() override;

private:
	// UE 5.3+에서는 TSharedPtr를 사용한 CPU 모델 인스턴스 관리가 표준입니다.
	TSharedPtr<UE::NNE::IModelInstanceCPU> ModelInstance;
};