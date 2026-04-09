// Fill out your copyright notice in the Description page of Project Settings.


#include "DataCollectComponent.h"

// Sets default values for this component's properties
UDataCollectComponent::UDataCollectComponent()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = false;

	// ...
}


// Called when the game starts
void UDataCollectComponent::BeginPlay()
{
	Super::BeginPlay();

	// ...
	
}

void UDataCollectComponent::SaveProjectileData(float Distance2D, float HeightDiff, float Angle)
{
    // 1. 저장 경로 설정 (프로젝트/Saved/ProjectileTrainingData.csv)
    FString FilePath = FPaths::ProjectSavedDir() + TEXT("ProjectileTrainingData.csv");

    // 2. 한 줄의 데이터 구성 (CSV 형식: 거리,높이차,각도,성공여부)
    // %f는 실수, %d는 정수(bool을 1 또는 0으로 변환)
    FString DataLine = FString::Printf(TEXT("%f,%f,%f,%d\n"),
        Distance2D,
        HeightDiff,
        Angle,
        Angle > 30 ? 1 : 0);

    // 3. 파일에 쓰기 (FILEWRITE_Append 옵션으로 기존 데이터 뒤에 계속 붙임)
    bool bSuccess = FFileHelper::SaveStringToFile(DataLine,
        *FilePath,
        FFileHelper::EEncodingOptions::AutoDetect,
        &IFileManager::Get(),
        FILEWRITE_Append);

    if (bSuccess)
    {
        UE_LOG(LogTemp, Log, TEXT("NNE Data Saved: %s"), *DataLine);
    }
}
