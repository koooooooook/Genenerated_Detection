#!/bin/bash

# whichfaceisreal/1_fake 폴더에서 모든 파일을 찾아서 반복 실행
gpu_ids=(0 1 2 3)  # 사용할 GPU ID 배열
index=0  # GPU 인덱스를 추적하는 변수

for img_path in ./dataset/sy_test_resized/dalle/1_fake/*
do
    img_file=$(basename "$img_path")
    gpu_id=${gpu_ids[$index]}  # 현재 이미지에 할당할 GPU ID
    echo "Processing $img_file on GPU $gpu_id, with index=$index"

    echo "$img_file"    
    CUDA_VISIBLE_DEVICES=$gpu_id python generator_finetuning_new.py --source dalle --ox 1_fake --img_name "$img_file" &

    
    # 다음 GPU ID로 업데이트
    index=$(( (index + 1) % 4 ))  # GPU ID 배열 길이에 맞게 인덱스를 순환

    # GPU 수만큼 실행 후 기다리기
    if [[ $((index % 4)) -eq    0 ]]; then
        wait
        fi
done                                    

wait  # 마지막 배치의 모든 프로세스가 끝나기를 기다림                                                                                                                                              
