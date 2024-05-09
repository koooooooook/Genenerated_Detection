import torch
from PIL import Image
import os
import torchvision.transforms as transforms

device = 'cuda'
data_root = '/home/sawyun.kyae/ftp_shared_data/dataset/CNNSpot/test'
source = 'dalle'
# classname = 'church'
ox = '1_fake'
img_dir = os.path.join(data_root, source, ox) # img_dir = os.path.join(data_root, source, classname, ox)
img_list = os.listdir(img_dir)

output_file = f'{source}_{ox}.csv' # output_file = f'{source}_{classname}_{ox}.csv'

# detector
det_arch='CLIP:ViT-L/14' # 'Imagenet:resnet18','Imagenet:resnet34','Imagenet:resnet50','Imagenet:resnet101','Imagenet:resnet152','Imagenet:vgg11','Imagenet:vgg19','Imagenet:swin-b','Imagenet:swin-s','Imagenet:swin-t','Imagenet:vit_b_16','Imagenet:vit_b_32','Imagenet:vit_l_16','Imagenet:vit_l_32','CLIP:RN50','CLIP:RN101','CLIP:RN50x4','CLIP:RN50x16','CLIP:RN50x64','CLIP:ViT-B/32','CLIP:ViT-B/16','CLIP:ViT-L/14','CLIP:ViT-L/14@336px',
det_model='UnivFD' # 'UnivFD', 'CNNDetection' 
crop_images = None # 224?

# START MAIN
device = torch.device(device)

# 디텍터
if det_model == 'UnivFD':
    from models_univdet import get_model
    det_ckpt = 'experiments/detector/fc_weights.pth'
    
    detector = get_model(det_arch)
    state_dict = torch.load(det_ckpt, map_location='cpu')
    detector.fc.load_state_dict(state_dict)
    print ("Univ detector loaded..")
    detector.eval()
    detector.cuda()
    detector = torch.nn.DataParallel(detector)

transform_univfd = transforms.Compose([
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize( mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711] ),
                ])

import csv
from tqdm import tqdm

with open(output_file, 'a', newline='') as file:
    writer = csv.writer(file)
    if file.tell() == 0:
        writer.writerow(['Source', 'ox', 'Img', 'Confidence Score']) # writer.writerow(['Source', 'Class', 'ox', 'Img', 'Confidence Score']) # 헤더 작성

    for target_image_path in tqdm(img_list, desc='Processing images'):
        img_path = os.path.join(img_dir, target_image_path)
        
        x = Image.open(img_path).convert("RGB")
        x = transform_univfd(x)
        x = x.unsqueeze(0)
        
        output = detector(x).cpu()
        prob = output.sigmoid().flatten()
        
        confidence = prob.item()
        
        # import pdb; pdb.set_trace()
        
        writer.writerow([source, ox, target_image_path, confidence]) # writer.writerow([source, classname, ox, target_image_path, confidence])

print("Confidence score is calculated and saved successfully.")

import pandas as pd
import shutil
from PIL import Image
import os
import matplotlib.pyplot as plt

# CSV 파일 읽기
# data_root = '/home/sawyun.kyae/ftp_shared_data/dataset/CNNSpot/test'
# source = 'stylegan2'
# classname = 'horse'
# ox = '1_fake'
# img_dir = os.path.join(data_root, source, classname, ox) # img_dir = os.path.join(data_root, source, ox) # img_dir = os.path.join(data_root, source, classname, ox)

file_dir = source + "_" +  ox + ".csv" # file_dir = source + "_" +  ox + ".csv" # file_dir = source + "_" + classname + "_" +  ox + ".csv"
df = pd.read_csv(file_dir)

# Confidence Score로 정렬
df_sorted = df.sort_values(by='Confidence Score', ascending=True)

# 상위 100개 및 하위 100개 선택
top100 = df_sorted.tail(100)
bottom100 = df_sorted.head(100)

top10 = df_sorted.tail(10)
bottom10 = df_sorted.head(10)
# 이미지 경로 및 점수 준비
image_paths = bottom10['Img'].tolist() + top10['Img'].tolist()
scores = bottom10['Confidence Score'].tolist() + top10['Confidence Score'].tolist()
titles = bottom10['Img'].tolist() + top10['Img'].tolist()  # 이미지 이름

# 이미지 저장 폴더 생성
output_folder = 'dataset/confidence_sorted/'
output_folder = os.path.join(output_folder, source, ox) # output_folder = os.path.join(output_folder, source, classname, ox)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 이미지 복사 및 이름 변경 함수
def copy_and_rename(image_df, img_dir, output_folder):
    for _, row in image_df.iterrows():
        # import pdb; pdb.set_trace()
        src_path = os.path.join(img_dir, row['Img'])
        confidence_score = f"{row['Confidence Score']:.2f}"
        file_extension = row['Img'].split('.')[-1]  # 확장자 추출
        new_filename = f"{confidence_score}_{row['Img'].split('.')[0]}.{file_extension}"
        dst_path = os.path.join(output_folder, new_filename)
        shutil.copy(src_path, dst_path)

def plot_images(image_paths, scores, titles, savefile, grid_dims=(4, 5), figsize=(20, 10)):
    fig, axes = plt.subplots(nrows=grid_dims[0], ncols=grid_dims[1], figsize=figsize)
    for ax, img_path, score, title in zip(axes.flatten(), image_paths, scores, titles):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(f"{title}\nScore: {score:.2f}", fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(savefile)
    plt.show()

# 이미지 복사 및 이름 변경 실행
copy_and_rename(top100, img_dir, output_folder)
copy_and_rename(bottom100, img_dir, output_folder)

print("Images have been sorted and saved successfully.")

plot_images(
    image_paths=[os.path.join(img_dir, img_name) for img_name in image_paths],
    scores=scores,
    titles=titles,
    savefile=file_dir.replace(".csv", ".png")
)

# 'Confidence Score' 열의 데이터로 히스토그램 생성
hist_name = file_dir.split('.')[0] + "_histogram.png"
plt.figure(figsize=(10, 6))
plt.hist(df['Confidence Score'], bins=[i/10.0 for i in range(11)], edgecolor='black', color='skyblue')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.title('Histogram of Confidence Scores')
plt.grid(True)
plt.xticks([i/10.0 for i in range(11)])  # x축 눈금 설정
plt.savefig(hist_name)  # 히스토그램 이미지 저장
plt.show()