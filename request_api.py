import requests
from io import BytesIO
import argparse
from pathlib import Path
import base64
from PIL import Image
import json

# parse
parser = argparse.ArgumentParser(description='sending to img2img api.')
parser.add_argument('-i', '--indir', default='./in/', type=str, help='input directory')
parser.add_argument('-m', '--mask', default='./mask/', type=str, help='input directory')
parser.add_argument('-o', '--outdir', type=str, default='./out/', help='out directory')

args = parser.parse_args()

files = Path(args.indir ).glob('*.png')

url = 'http://127.0.0.1:7860/sdapi/v1/img2img'
for file in files:
    with open(args.indir + file.stem + ".png", "rb") as image_file:
        img = base64.b64encode(image_file.read())
    with open(args.mask + file.stem + ".png", "rb") as image_file:
        mask = base64.b64encode(image_file.read())
    # api options (there are more options available)
    dics = {
        # must be a list
        "init_images": [img],
        'mask':mask,
        'prompt':'high resolution cyberpunk robotic',
        'negative_prompt': 'cgi',
        'steps': 30,
        # 0 does not invert, 1 does invert 
        'inpainting_mask_invert':0,
        # 0 black, 1 original , 2 latent space
        'inpainting_fill':1,
        'sampler_name': 'Euler a', 
        'cfg_scale': 7, 
        'seed': 316429, 
        'width': 768,
        'height': 1024, 
        'denoising_strength': 0.6, 
        'mask_blur': 2
    }
    x = requests.post(url, json = dics)
    img1 = json.loads(x.text)['images'][0]
    im = Image.open(BytesIO(base64.b64decode(img1)))
    #saving image
    im.save(args.outdir + file.stem + ".png", 'PNG')
