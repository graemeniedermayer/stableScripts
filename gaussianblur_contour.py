from PIL import Image
from pathlib import Path
import cv2
import numpy as np
import imageio


#load files
files = sorted(Path("./" ).glob('*.png'))

for file in files:
    input_path = str(file)

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    # Convert to bgr if colored image
    # img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

    #set a thresh
    thresh = 10
    #get threshold image
    ret, thresh_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    #find contours
    contours, hierarchy = cv2.findContours(thresh_img.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #create an empty image for contours
    img_contours = np.zeros(img.shape)
    outer_contours = np.zeros(img.shape)
    # make mask
    cv2.drawContours(img_contours, contours, -1, (255,255,255), 5)
    cv2.drawContours(outer_contours, contours, -1, (255,255,255), -1)
    blur = cv2.GaussianBlur(img, (17,17), 3)

    im = np.array(Image.open(input_path)).astype("uint16")

    im[img_contours>0] = blur[img_contours>0]

    imageio.imwrite("gaus/" + file.stem + '.png',im) 
