// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "DataCollectComponent.generated.h"


UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class PROJECTNNE_API UDataCollectComponent : public UActorComponent
{
	GENERATED_BODY()

public:	
	// Sets default values for this component's properties
	UDataCollectComponent();

protected:
	// Called when the game starts
	virtual void BeginPlay() override;

public:
	/** * 데이터를 CSV 파일로 저장하는 함수
	 * @param Distance2D 수평 거리
	 * @param HeightDiff 높이 차이 (Target Z - Start Z)
	 * @param Angle      발사 각도
	 */
	UFUNCTION(BlueprintCallable, Category = "NNE|Data")
	void SaveProjectileData(float Distance2D, float HeightDiff, float Angle);

		
};
