// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
// --- NNE 시스템 핵심 헤더 3종 세트 ---
#include "NNE.h"             // NNE 기본 인터페이스
#include "NNETypes.h"        // ★ 추가: FTensorBindingCPU 등 타입 정의
#include "NNERuntimeCPU.h"   // CPU 실행을 위한 인터페이스
#include "NNEModelData.h"    // 모델 애셋 데이터 타입
// ------------------------------------
#include "NNE_SimpleDouble.generated.h"

UCLASS()
class PROJECTNNE_API ANNE_SimpleDouble : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ANNE_SimpleDouble();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	// 에디터에서 할당할 모델 파일 (.onnx)
	UPROPERTY(EditAnywhere, Category = "NNE")
	TObjectPtr<UNNEModelData> ModelData;

	// 실제 연산을 수행하는 두뇌 (인스턴스)
	TSharedPtr<UE::NNE::IModelInstanceCPU> ModelInstance;

	UFUNCTION(BlueprintImplementableEvent, Category = "NNE")
	void OnModelReady();

public:	
	// 블루프린트에서 호출할 함수 (입력값 x 2 결과를 반환)
	UFUNCTION(BlueprintCallable, Category = "NNE")
	float PredictDouble(float InputValue);

private:
	// 모델 로드 및 초기화 로직
	bool InitializeNNE();

};
