// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "NNE_DataCollector.generated.h"

UCLASS()
class PROJECTNNE_API ANNE_DataCollector : public AActor
{
    GENERATED_BODY()

public:
    ANNE_DataCollector();

    // 저장할 파일 이름 (예: ProjectileData.csv)
    UPROPERTY(EditAnywhere, Category = "NNE|Data")
    FString FileName = TEXT("ProjectileTrainingData.csv");

    /** * 데이터를 수집하여 파일에 한 줄씩 추가합니다.
     * 블루프린트에서 투사체가 타겟에 맞거나 파괴되는 시점에 호출하면 됩니다.
     */
    UFUNCTION(BlueprintCallable, Category = "NNE|Data")
    void LogProjectileData(FVector StartLoc, FVector TargetLoc, float Angle, float Impulse, float Weight, float Radius, float TargetRad, bool bHit);

private:
    // 파일에 문자열을 쓰는 헬퍼 함수
    void SaveToFile(FString DataLine);
};