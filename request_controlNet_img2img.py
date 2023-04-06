import requests
from io import BytesIO
import argparse
from pathlib import Path
import base64
from PIL import Image
import json

# parse
parser = argparse.ArgumentParser(description='sending to img2img api.')
parser.add_argument('-i1', '--indir1', default='./in1/', type=str, help='input directory')
parser.add_argument('-i2', '--indir2', default='./in2/', type=str, help='input directory')
parser.add_argument('-m', '--mask', default='./mask/', type=str, help='input directory')
parser.add_argument('-o', '--outdir', type=str, default='./out/', help='out directory')

conv = {
}

args = parser.parse_args()

import glob
import os
init_files = Path(args.indir1 ).glob('*.png')
init_files = sorted(init_files)
controlnet_files = Path(args.indir2 ).glob('*.png')
controlnet_files = sorted(controlnet_files, key=os.path.getmtime)

# there are a bunch of other api requests available
# it will evaluate using the model currently loaded 
url = 'http://127.0.0.1:7860/controlnet/img2img'
for init, controlnet in zip(init_files, controlnet_files):

    with open(args.indir1 +'/' + init.stem + ".png", "rb") as image_file:
        init_img = base64.b64encode(image_file.read())
    with open(args.indir2 +'/'+ controlnet.stem + ".png", "rb") as image_file:
        ctl_img = base64.b64encode(image_file.read())
    print(controlnet.stem)
    print(init.stem)
    #with open(args.mask + file.stem + ".png", "rb") as image_file:
    #    mask = base64.b64encode(image_file.read())
    # api options (there are more options available)
    dics = {
    
  	"init_images": [
    	init_img
  	],
  	"resize_mode": 0,
  	"denoising_strength": 0.55,
        'prompt':"",
        # must be a list
        # 'mask':mask,
        'negative_prompt':"",
        "seed": 345,
        "subseed": -1,
        "subseed_strength": 0.2,
        "batch_size": 1,
        "n_iter": 1,
        "steps": 25,
        "cfg_scale": 7,
        "width": 512,
        "height": 512,#896,
        "restore_faces": True,
        "eta": 0,
        "controlnet_input_image": [ctl_img],
        "controlnet_module": 'openpose',
        "controlnet_model": 'control_sd15_openpose [fef5e48e]',
        "controlnet_weight": 1.0,
        "controlnet_guidance": 1.0,
    }
    x = requests.post(url, json = dics)
    img1 = json.loads(x.text)['images'][0]
    im = Image.open(BytesIO(base64.b64decode(img1)))
    #saving image
    im.save(args.outdir +'/'+ controlnet.stem + ".png", 'PNG')
