from super_image import EdsrModel, ImageLoader
from PIL import Image
import requests
import time

url = 'https://upload.wikimedia.org/wikipedia/commons/2/25/Blisk-logo-512-512-background-transparent.png?20160517154140'
image = Image.open(requests.get(url, stream=True).raw)

# Start timing
start_time = time.time()

model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
inputs = ImageLoader.load_image(image)
preds = model(inputs)

# End timing
end_time = time.time()
processing_time = end_time - start_time
print(f"Upscaling took {processing_time:.2f} seconds")

ImageLoader.save_image(preds, './scaled_2x.png')
ImageLoader.save_compare(inputs, preds, './scaled_2x_compare.png')