# largely paraphrased from
# https://google.github.io/mediapipe/
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# For static images:
BG_COLOR = (0, 0, 0) # background colour
MASK_COLOR = (255, 255, 255) # masking colour
#hard coded input directory
output_path = './pictures'
#load files
files = Path( output_path ).glob('*.png')

with mp_selfie_segmentation.SelfieSegmentation(
    model_selection=0) as selfie_segmentation:
  for idx, file in enumerate(files):
    image = cv2.imread(output_path + '/' + file.stem + '.png')
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw selfie segmentation on the background image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    # Generate solid color images for showing the output selfie segmentation mask.
    fg_image = np.zeros(image.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    output_image = np.where(condition, fg_image, bg_image)
    cv2.imwrite(str(idx) + '.png', output_image)
