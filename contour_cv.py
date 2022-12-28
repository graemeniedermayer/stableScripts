from PIL import Image
import argparse
from pathlib import Path
import cv2
import numpy as np

# parser
parser = argparse.ArgumentParser(description='Cut-out significant contours.')
parser.add_argument('-i', '--indir', default='./', type=str, help='input directory')
parser.add_argument('-o', '--outdir', type=str, default='./', help='out directory')

args = parser.parse_args()

#load files
files = Path(args.indir ).glob('*.png')

for file in files:
    input_path = str(file)

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    #convert img to grey
    img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #set a thresh
    thresh = 100
    #get threshold image
    ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    #find contours
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #create an empty image for contours
    img_contours = np.zeros(img.shape)
    # make mask
    cv2.drawContours(img_contours, contours, -1, (255,255,255), -1)
    #invert mask
    img_contours = -1*(img_contours-255)
    im = Image.open(input_path)
    
    im.putalpha(255)
    #mask contour to zero
    imArr = np.array(im)
    imArr[:,:,:3][img_contours[:,:,:3]>10] = 0
    imArr[img_contours[:,:,0]>10,3] = 0
    PIL_image = Image.fromarray(imArr.astype('uint8'), 'RGBA')   
    # resizing example line
    # PIL_image = PIL_image.resize((800,800), Image.ANTIALIAS)

    #saving image
    PIL_image.save(args.outdir + file.stem + '.out.png', 'PNG')