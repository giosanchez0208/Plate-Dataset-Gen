import os
import cv2
import numpy as np

IMAGES_DIR = 'dataset/images'
MASKS_DIR = 'dataset/masks'

def get_plate_ids():
    return sorted([
        f.replace('plate_', '').replace('.jpg', '')
        for f in os.listdir(IMAGES_DIR) if f.startswith('plate_') and f.endswith('.jpg')
    ])

def get_mask_paths(plate_id):
    return sorted([
        os.path.join(MASKS_DIR, f)
        for f in os.listdir(MASKS_DIR)
        if f.startswith(f'plate_{plate_id}_char_') and f.endswith('.png')
    ], key=lambda p: int(p.split('_char_')[-1].split('.')[0]))

def load_overlay(image_path, mask_paths):
    img = cv2.imread(image_path)
    overlay = img.copy()

    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.shape[:2] == overlay.shape[:2]:
            overlay[mask > 0] = [0, 0, 255]

    return overlay

def run_viewer():
    plate_ids = get_plate_ids()
    if not plate_ids:
        print("No plates found.")
        return

    idx = 0
    while True:
        plate_id = plate_ids[idx]
        img_path = os.path.join(IMAGES_DIR, f'plate_{plate_id}.jpg')
        mask_paths = get_mask_paths(plate_id)

        if not os.path.exists(img_path):
            print(f"[MISSING] {img_path}")
            idx = (idx + 1) % len(plate_ids)
            continue

        print(f"[{idx+1}/{len(plate_ids)}] plate_{plate_id} - {len(mask_paths)} masks")
        overlay = load_overlay(img_path, mask_paths)
        cv2.imshow('Mask Checker - A/D to scroll, ESC to quit', overlay)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('d'):
            idx = (idx + 1) % len(plate_ids)
        elif key == ord('a'):
            idx = (idx - 1 + len(plate_ids)) % len(plate_ids)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_viewer()
