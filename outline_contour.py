from PIL import Image
from pathlib import Path
import cv2
import numpy as np


files = Path('./').glob('*.png')

for file in files:
    input_path = str(file)

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    #convert img to grey for rgb image (remove for greyscale)
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
    
    f = file.stem
    
    im.putalpha(255)
    #mask contour to zero
    imArr = np.array(im)
    imArr[:,:,:3][img_contours[:,:,:3]>10] = 0
    imArr[img_contours[:,:,0]>10,3] = 0
    PIL_image = Image.fromarray(imArr.astype('uint8'), 'RGBA')   
    PIL_image.save(f+'.out.png', 'PNG')
