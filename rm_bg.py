
from PIL import Image
import numpy as np
import argparse
from pathlib import Path
from rembg import remove, new_session

parser = argparse.ArgumentParser(description='remove background.')
parser.add_argument('-i', '--indir', default='./in', type=str, help='input directory')
parser.add_argument('-o', '--outdir', type=str, default='./bot/', help='out directory')
parser.add_argument('-t', '--topdir', type=str, default='./top/', help='out directory')
parser.add_argument('-m', '--model', type=str, default='u2net', help='models')

args = parser.parse_args()

session = new_session(args.model)

files = Path(args.indir ).glob('*.png')

for file in files:
    input_path = str(file)
    output_path = args.outdir
    input = Image.open(input_path)
    output = remove(input, session=session)
    # model based output
    if 'u2net_cloth_seg' != args.model:
        output.save(output_path + file.stem + ".out.png")
    else:
        split = np.array(output)
        hsplit = int(split.shape[0]/3)
        #spliting single image into the 3 clothing types
        top = split[:hsplit, :] 
        bot = split[hsplit:2*hsplit,:]
        comb =  split[2*hsplit:,:]
        # example of boolean masking and casting
        bot[(bot[:,:,0]>1) | (bot[:,:,1]>1) | (bot[:,:,2]>1)] = 255
        top[(top[:,:,0]>1) | (top[:,:,1]>1) | (top[:,:,2]>1)] = 255
        comb[(comb[:,:,0]>1) | (comb[:,:,1]>1) | (comb[:,:,2]>1)] = 255
        if np.sum(bot)>np.sum(comb):
            Image.fromarray(bot).save(output_path + file.stem + '.png')
        else:
        	Image.fromarray(comb).save(output_path + file.stem + '.png')

            
