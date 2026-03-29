// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "NNE.h"
#include "NNERuntimeCPU.h"
#include "NNEModelData.h"
#include "NNETypes.h"
#include "NNE_Runner.generated.h"

UCLASS()
class PROJECTNNE_API ANNE_Runner : public AActor
{
    GENERATED_BODY()

public:
    ANNE_Runner();

protected:
    virtual void BeginPlay() override;

    /** NNE 초기화를 담당하는 내부 함수들 (모듈화) */
    bool InitializeNNE();
    bool CreateModelInstance();

public:
    /** 모델 에셋 */
    UPROPERTY(EditAnywhere, Category = "NNE")
    TObjectPtr<UNNEModelData> ModelData;

    /** 텐서 설정 (에디터에서 수정 가능) */
    UPROPERTY(EditAnywhere, Category = "NNE|Settings")
    int32 BatchSize;

    UPROPERTY(EditAnywhere, Category = "NNE|Settings")
    int32 FeatureCount;

    /** 초기화 완료 시 블루프린트로 알림 */
    UFUNCTION(BlueprintImplementableEvent, Category = "NNE")
    void OnModelInitialized();

    /** 추론 함수 (1개 입력용 - ModelTools) */
    UFUNCTION(BlueprintCallable, Category = "NNE")
    float PredictDouble(float InputValue);

    /** 추론 함수 (2개 입력용 - TrainMulti) */
    UFUNCTION(BlueprintCallable, Category = "NNE")
    float PredictDamage(float Strength, float Skill);

    /** 추론 함수 (입력은 하나지만 결과는 여러 개인 경우 - TrainMultiOutput) */
    UFUNCTION(BlueprintCallable, Category = "NNE")
    TArray<float> PredictMultiple(float InputValue);

private:
    /** NNE 모델 인스턴스 포인터 */
    TSharedPtr<UE::NNE::IModelInstanceCPU> ModelInstance;
};