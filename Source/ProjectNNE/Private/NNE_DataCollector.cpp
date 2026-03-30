// Fill out your copyright notice in the Description page of Project Settings.


#include "NNE_DataCollector.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

ANNE_DataCollector::ANNE_DataCollector() { PrimaryActorTick.bCanEverTick = false; }

void ANNE_DataCollector::LogProjectileData(FVector StartLoc, FVector TargetLoc, float Angle, bool bHit)
{
    // 1. 위치 차이 계산 (벡터 뺄셈)
    FVector Diff = TargetLoc - StartLoc;

    // 2. 수평 거리 (XY 평면상의 거리)
    float HorizontalDist = FVector2D(Diff.X, Diff.Y).Size();

    // 3. 높이 차이 (Z축 차이)
    float HeightDiff = Diff.Z;

    // 4. CSV 파일 구성 (총 4개 열로 축소)
    // 인덱스: [0]수평거리, [1]높이차, [2]각도, [3]성공여부
    FString DataRow = FString::Printf(TEXT("%f,%f,%f,%d\n"),
        HorizontalDist,
        HeightDiff,
        Angle,
        bHit ? 1 : 0
    );

    // 5. 경로 설정: 프로젝트의 Saved 폴더 + FileName
    FString FullPath = FPaths::ProjectSavedDir() + FileName;

    // 6. 파일 저장 로직
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();

    // 파일이 있으면 추가(Append), 없으면 생성(Write)
    FFileHelper::SaveStringToFile(DataRow, *FullPath, FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(), FILEWRITE_Append);
}

void ANNE_DataCollector::SaveToFile(FString DataLine)
{
    // 프로젝트의 Saved/Logs 폴더에 저장됩니다.
    FString FullPath = FPaths::ProjectSavedDir() + FileName;

    // FILEWRITE_Append 옵션을 사용하여 기존 데이터 뒤에 계속 이어 붙입니다.
    FFileHelper::SaveStringToFile(DataLine, *FullPath, FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(), EFileWrite::FILEWRITE_Append);
}