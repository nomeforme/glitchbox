import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from skimage.io import imread
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
import requests
import io
import os
import time
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation

model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4",trust_remote_code=True)

# # Compile the model if torch.compile is available
# if hasattr(torch, 'compile'):
#     print("Compiling model...")
#     model = torch.compile(model)

def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    # orig_im_size=im.shape[0:2]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), size=model_input_size, mode='bilinear')
    image = torch.divide(im_tensor,255.0)
    image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])
    return image

def postprocess_image(result: torch.Tensor, im_size: list)-> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear') ,0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result-mi)/(ma-mi)
    im_array = (result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

model_input_size = [1024, 1024]

# prepare input
image_path = "https://farm5.staticflickr.com/4007/4322154488_997e69e4cf_z.jpg"
response = requests.get(image_path)
orig_im = imread(image_path)
orig_im_size = orig_im.shape[0:2]
image = preprocess_image(orig_im, model_input_size).to(device)

# inference 
t1 = time.time()   
result=model(image)
t2 = time.time()
print(f"Inference time: {t2 - t1} seconds")

# post process
result_image = postprocess_image(result[0][0], orig_im_size)

# save result
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'no_bg_image.png')
pil_mask_im = Image.fromarray(result_image)
orig_image = Image.open(io.BytesIO(response.content))
no_bg_image = orig_image.copy()
no_bg_image.putalpha(pil_mask_im)
no_bg_image.save(output_path)

# Initialize SelfiSegmentation
segmentor = SelfiSegmentation()

# Read image using OpenCV
response = requests.get(image_path)
image_data = np.asarray(bytearray(response.content), dtype=np.uint8)
imgOffice = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

# Resize image to 640x480
imgOffice = cv2.resize(imgOffice, (640, 480))

# Define green color for background
green = (0, 255, 0)

# Remove background
imgNoBg = segmentor.removeBG(imgOffice, green)

# Save the result using OpenCV
output_path_cv = os.path.join(output_dir, 'no_bg_image_cv.png')
cv2.imwrite(output_path_cv, imgNoBg)
