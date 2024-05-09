# python preprocess.py --img_path /mnt/shared/dataset/CNNSpot/test/confidence_sorted

# #!/bin/bash
# gpu_ids=(0 1 2 3)  # 사용할 GPU ID 배열
# index=0  # GPU 인덱스를 추적하는 변수

# # python preprocess.py --img_path /mnt/shared/dataset/CNNSpot/test/confidence_sorted

# for img_path in $(find ./dataset -type f); do
#     img_file=$(basename "$img_path")
#     gpu_id=${gpu_ids[$index]}  # 현재 이미지에 할당할 GPU ID
#     echo "(index=$index) GPU $gpu_id : Processing $img_file"

#     CUDA_VISIBLE_DEVICES=$gpu_id python generator_training.py --img_path "$img_path" &
    
#     # 다음 GPU ID로 업데이트
#     index=$(( (index + 1) % 4 ))  # GPU ID 배열 길이에 맞게 인덱스를 순환

#     # GPU 수만큼 실행 후 기다리기
#     if [[ $((index % 4)) -eq    0 ]]; then
#         wait
#         fi
# done

id=0

CUDA_VISIBLE_DEVICES=$id python generator_training.py --config configuration.yaml --training_list dataset/training_list$id.txt