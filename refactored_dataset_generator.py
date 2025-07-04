import os
import random
time_import = __import__('time')
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from refactored_plate_generator import generate_random_plate

# Directories
BACKGROUND_DIR = 'backgrounds'
OUTPUT_DIR = 'augmented_dataset'
IMG_OUT = os.path.join(OUTPUT_DIR, 'images')
LBL_OUT = os.path.join(OUTPUT_DIR, 'labels')

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(LBL_OUT, exist_ok=True)

# Config
CANVAS_SIZE = (512, 512)

# Helper to load random background
def load_random_background():
    files = [f for f in os.listdir(BACKGROUND_DIR) if f.lower().endswith(('jpg','png','jpeg'))]
    bg = Image.open(os.path.join(BACKGROUND_DIR, random.choice(files))).convert('RGB')
    return bg.resize(CANVAS_SIZE, Image.ANTIALIAS)

# Apply random affine + perspective
def transform_plate_and_masks(plate, masks):
    w, h = CANVAS_SIZE
    # combine plate and masks into arrays
    plate_arr = np.array(plate)
    mask_arrs = [(ch, np.array(mask)) for ch, mask in masks]

    # random scale
    scale = random.uniform(0.5, 1.0)
    tw, th = int(w * scale), int(h * scale / 4)
    # random rotation
    angle = random.uniform(-30, 30)
    # random translation
    tx = random.randint(0, w - tw)
    ty = random.randint(0, h - th)

    # resize plate
    plate_small = plate.resize((tw, th), Image.ANTIALIAS)
    # transform plate
    M = cv2.getRotationMatrix2D((tw/2, th/2), angle, 1)
    plate_rot = cv2.warpAffine(np.array(plate_small), M, (tw, th), flags=cv2.INTER_LINEAR)

    # perspective can be added here if desired

    # position on canvas
    canvas_img = load_random_background()
    canvas_arr = np.array(canvas_img)
    canvas_arr[ty:ty+th, tx:tx+tw] = plate_rot

    # transform masks similarly
    transformed_masks = []
    for ch, mask in mask_arrs:
        mask_small = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
        mask_rot = cv2.warpAffine(mask_small, M, (tw, th), flags=cv2.INTER_NEAREST)
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[ty:ty+th, tx:tx+tw] = mask_rot
        transformed_masks.append((ch, full_mask))

    return canvas_arr, transformed_masks

# Generate annotation in YOLO format
def generate_yolo_annotations(masks, image_shape):
    h, w = image_shape[:2]
    lines = []
    for ch, mask in masks:
        ys, xs = np.where(mask > 0)
        if len(xs)==0: continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        cx = ((x0 + x1)/2) / w
        cy = ((y0 + y1)/2) / h
        cw = (x1 - x0) / w
        ch_h = (y1 - y0) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {cw:.6f} {ch_h:.6f}")
    return lines

if __name__ == '__main__':
    idx = 0
    while True:
        plate, masks = generate_random_plate()
        canvas, t_masks = transform_plate_and_masks(plate, masks)

        cv2.imshow('Augmented Plate', cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(0)
        if key != 32: break

        # Save
        img_name = f'aug_{idx:05d}.jpg'
        cv2.imwrite(os.path.join(IMG_OUT, img_name), canvas)
        ann = generate_yolo_annotations(t_masks, canvas.shape)
        with open(os.path.join(LBL_OUT, img_name.replace('.jpg','.txt')), 'w') as f:
            f.write('\n'.join(ann))
        idx += 1

    cv2.destroyAllWindows()
