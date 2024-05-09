from PIL import Image
import os
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# shared_data/dataset/CNNSpot/test/confidence_sorted/whichfaceisreal/0_real/0.00_00012.jpeg
# shared_data/dataset/CNNSpot/test/confidence_sorted/whichfaceisreal/1_fake/0.02_image-2019-02-17_022542.jpeg
def resize_jpg_to_png(directory):
    for root, dirs, files in os.walk(directory):
        for file in tqdm(files):
            file_path = os.path.join(root, file)
            # print(file_path) /mnt/shared/dataset/CNNSpot/test/confidence_sorted/stylegan2/car/0_real/0.10_00941.png
            # Iterate over each file in the directory
            if file_path.endswith(".jpg") or file_path.endswith(".jpeg") or file_path.endswith(".png"):
                # Open the image file
                image = Image.open(file_path)

                # Resize the image to 256x256 using the ANTIALIAS resampling filter
                # Use LANCZOS or Resampling.LANCZOS instead.
                # resized_image = image.resize((256, 256))
                resized_image = image.resize((256, 256), Image.ANTIALIAS)
            else:
                continue
            
            filename    = ''.join(os.path.splitext(os.path.basename(file_path))[0].split('_')[1:])
            data_class  = file_path.split('/')[-3]
            tag         = file_path.split('/')[-2].split('_')[-1]
            new_filename = f"{data_class}_{filename}"
            
            # Save the resized image with a new filename in PNG format
            dataset_dir = os.path.join('./dataset', tag)
            new_file_dir = os.path.join(dataset_dir, f'{new_filename}.png')
            os.makedirs(dataset_dir, exist_ok=True)
            if not os.path.exists(new_file_dir):
                resized_image.save(new_file_dir, "PNG")
                with open('./dataset/training_list.txt', 'a') as f:
                    f.write(f"{new_file_dir}\n")

            # Close the image file
            image.close()

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, help="dataset folder", default='/mnt/shared/dataset/CNNSpot/test/confidence_sorted')
args = parser.parse_args()

resize_jpg_to_png(args.img_path)