from PIL import Image

# Open the image file
# filename = '/home/junwon.ko/gen_det/dataset/real/real_cat_0000.png'
# image = Image.open(filename)

root = './sy_test/'
dest = './sy_test_resized/'

resol = 256

import os
     
def crop(image):
# Calculate the dimensions for center cropping
    width, height = image.size
    
    if width == height:
        return image.resize((256, 256))
    
    left = (width - resol) // 2
    top = (height - resol) // 2
    right = left + resol
    bottom = top + resol

    # # Crop the image
    cropped_image = image.crop((left, top, right, bottom))

    # import torchvision.transforms as transforms
    # transformed = transforms.Compose([
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
    # ])

    return cropped_image

# # Display the cropped image
# cropped_image.save(f"{filename.split('.png')[0]}_{resol}.png", 'PNG')

# # Close the image file
# image.close()

for subdir, dirs, files in os.walk(root):
    for file in files:
        # 파일 경로 생성
        src_path = os.path.join(subdir, file)
        target_path = os.path.join(subdir.replace(root, dest), file)
        
        # 이미지 파일인지 확인
        if src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 이미지 열기
            with Image.open(src_path) as img:
                cropped = crop(img)
                
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                cropped.save(target_path)
            