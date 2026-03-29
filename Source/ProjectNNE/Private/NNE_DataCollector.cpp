// Fill out your copyright notice in the Description page of Project Settings.


#include "NNE_DataCollector.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

ANNE_DataCollector::ANNE_DataCollector() { PrimaryActorTick.bCanEverTick = false; }

void ANNE_DataCollector::LogProjectileData(FVector StartLoc, FVector TargetLoc, float Angle, float Impulse, float Weight, float Radius, float TargetRad, bool bHit)
{
    // 1. 위치 차이 계산 (벡터 뺄셈)
    FVector Diff = TargetLoc - StartLoc;

    // 2. 수평 거리 (XY 평면상의 거리)
    float HorizontalDist = FVector2D(Diff.X, Diff.Y).Size();

    // 3. 높이 차이 (Z축 차이: 타겟이 높으면 +, 낮으면 -)
    float HeightDiff = Diff.Z;

    // 4. CSV 파일에 쓸 내용 구성 (항목 순서 중요!)
    // 이제 첫 번째 열은 수평거리, 두 번째 열은 높이차이가 됩니다.
    FString DataRow = FString::Printf(TEXT("%f,%f,%f,%f,%f,%f,%f,%d\n"),
        HorizontalDist, // [0] 수평 거리
        HeightDiff,     // [1] 높이 차이
        Angle,          // [2] 각도
        Impulse,        // [3] 힘
        Weight,         // [4] 무게
        Radius,         // [5] 반지름
        TargetRad,      // [6] 타겟 반지름
        bHit ? 1 : 0    // [7] 성공 여부
    );

    // 5. 경로 설정: 프로젝트의 Saved 폴더 + FileName
    FString FullPath = FPaths::ProjectSavedDir() + FileName;

    // 파일 저장 로직 (기존 코드 유지)
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    if (PlatformFile.FileExists(*FullPath))
    {
        FFileHelper::SaveStringToFile(DataRow, *FullPath, FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(), FILEWRITE_Append);
    }
    else
    {
        // 파일이 없으면 헤더와 함께 새로 생성 (선택 사항)
        FFileHelper::SaveStringToFile(DataRow, *FullPath);
    }
}

void ANNE_DataCollector::SaveToFile(FString DataLine)
{
    // 프로젝트의 Saved/Logs 폴더에 저장됩니다.
    FString FullPath = FPaths::ProjectSavedDir() + FileName;

    // FILEWRITE_Append 옵션을 사용하여 기존 데이터 뒤에 계속 이어 붙입니다.
    FFileHelper::SaveStringToFile(DataLine, *FullPath, FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(), EFileWrite::FILEWRITE_Append);
}