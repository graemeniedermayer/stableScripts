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

# there are a bunch of other api requests available
# it will evaluate using the model currently loaded 
url = 'http://127.0.0.1:7860/controlnet/txt2img'
for file in files:
    with open(args.indir + file.stem + ".png", "rb") as image_file:
        img = base64.b64encode(image_file.read())
    #with open(args.mask + file.stem + ".png", "rb") as image_file:
    #    mask = base64.b64encode(image_file.read())
    # api options (there are more options available)
    dics = {
        # must be a list
        # 'mask':mask,
        'prompt':"",
        'negative_prompt':"",
        "seed": 345,
        "subseed": -1,
        "subseed_strength": 0.1,
        "batch_size": 1,
        "n_iter": 1,
        "steps": 20,
        "cfg_scale": 7,
        "width": 512,
        "height": 512,#896,
        "restore_faces": True,
        "eta": 0,
        "sampler_index": "Euler a",
        "controlnet_input_image": [img],
        "controlnet_module": 'openpose',
        "controlnet_model": 'control_sd15_openpose [fef5e48e]',
        "controlnet_weight": 1.0,
        "controlnet_guidance": 1.0,
    }
    x = requests.post(url, json = dics)
    img1 = json.loads(x.text)['images'][0]
    im = Image.open(BytesIO(base64.b64decode(img1)))
    #saving image
    im.save(args.outdir + file.stem + ".png", 'PNG')
