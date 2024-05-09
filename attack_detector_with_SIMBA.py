import torch
from PIL import Image
import os
import torchvision.transforms as transforms

device = 'cuda'
data_root = './dataset/sy_test_resized/'
source = 'stylegan2'
ox = '1_fake'
img_dir = os.path.join(data_root, source, ox)
img_list = os.listdir(img_dir)

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
    
elif det_model == 'CNNDetection':
    from models_CNNDetection.demo_output import get_model
    det_ckpt = 'experiments/detector/blur_jpg_prob0.5.pth'

    detector = get_model()
    state_dict = torch.load(det_ckpt, map_location='cpu')
    detector.load_state_dict(state_dict['model'])
    detector.cuda()
    detector.eval()
    detector = torch.nn.DataParallel(detector)

from simba.simba import SimBA

# target_image_path = 'image-2019-02-16_235902.jpeg' #gen_dalle_aahxgrlfqt {train_with}_face_0000.png
for target_image_path in img_list:
    
    save_dir = img_dir.replace('sy_test_resized', 'sy_test_resized_attacked')
    save_image_path = os.path.join(save_dir, target_image_path)
    os.makedirs(os.path.dirname(save_image_path), exist_ok=True)

    img_path = os.path.join(img_dir, target_image_path)
    print("attacking image path: ", os.path.join(source, ox, target_image_path))

    target_image = Image.open(img_path).convert("RGB")
    transform_univfd = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize( mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711] ),
            ])
    target_image = transform_univfd(target_image)

    attacker = SimBA(detector, "CLIP", target_image.shape)
    labels = torch.ones(1)
    import time
    start_time = time.time()
    x = attacker.simba_single(target_image, labels) # [1, 3, 224, 224], [1.]
    print(time.time() - start_time)

    def tensor_to_pil(tensor):
        tensor = tensor.clone()  # 복사본을 만들어 원본 데이터를 보존
        # tensor = tensor * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
        tensor = tensor.clamp(0, 1)  # 값의 범위를 0과 1 사이로 조정
        # PIL 이미지로 변환
        image = transforms.ToPILImage()(tensor)
        return image

    # target_image를 PIL 이미지로 변환
    pil_image = tensor_to_pil(x)

    # 이미지 파일로 저장
    pil_image.save(save_image_path)