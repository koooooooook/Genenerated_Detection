from PIL import Image

# Open the image file
filename = '/home/junwon.ko/gen_det/dataset/real/real_cat_0000.png'
image = Image.open(filename)

resol = 224

# Calculate the dimensions for center cropping
width, height = image.size
left = (width - resol) // 2
top = (height - resol) // 2
right = left + resol
bottom = top + resol

# Crop the image
cropped_image = image.crop((left, top, right, bottom))

import torchvision.transforms as transforms
transformed = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
])

# Display the cropped image
cropped_image.save(f"{filename.split('.png')[0]}_{resol}.png", 'PNG')

# Close the image file
image.close()