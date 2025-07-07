import os
import random
time_import = __import__('time')
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from refactored_plate_generator import generate_random_plate

BACKGROUND_DIR = 'backgrounds'
OUTPUT_DIR = 'augmented_dataset'
IMG_OUT = os.path.join(OUTPUT_DIR, 'images')
LBL_OUT = os.path.join(OUTPUT_DIR, 'labels')

CANVAS_SIZE = (512, 512)
SCALE_RANGE = (0.15, 0.4)
ROTATION_RANGE = (-45, 45)
PERSPECTIVE_STRENGTH = 0.3
SHEAR_RANGE = (-0.3, 0.3)
BLUR_RANGE = (0, 2)
NOISE_INTENSITY = 0.1
BRIGHTNESS_RANGE = (0.7, 1.3)
CONTRAST_RANGE = (0.8, 1.2)
PADDING = 60
PERSPECTIVE_PROB = 0.9
SHEAR_PROB = 0.8
BLUR_PROB = 0.6
NOISE_PROB = 0.7

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(LBL_OUT, exist_ok=True)

def load_random_background():
    files = [f for f in os.listdir(BACKGROUND_DIR) if f.lower().endswith(('jpg','png','jpeg'))]
    bg = Image.open(os.path.join(BACKGROUND_DIR, random.choice(files))).convert('RGB')
    return bg.resize(CANVAS_SIZE, Image.LANCZOS)

def apply_perspective(img, strength=PERSPECTIVE_STRENGTH):
    h, w = img.shape[:2]
    
    dx = random.uniform(-strength, strength) * w
    dy = random.uniform(-strength, strength) * h
    
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([
        [dx, dy],
        [w - dx, dy], 
        [w - dx, h - dy],
        [dx, h - dy]
    ])
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

def apply_shear(img, shear_x, shear_y):
    h, w = img.shape[:2]
    M = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

def add_blur(img):
    if random.random() < BLUR_PROB:
        kernel_size = random.randint(1, BLUR_RANGE[1]) * 2 + 1
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return img

def add_noise(img):
    if random.random() < NOISE_PROB:
        noise = np.random.normal(0, NOISE_INTENSITY * 255, img.shape).astype(np.uint8)
        return cv2.add(img, noise)
    return img

def adjust_brightness_contrast(img):
    alpha = random.uniform(*CONTRAST_RANGE)
    beta = random.uniform(*BRIGHTNESS_RANGE) * 100 - 100
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# Apply random affine + perspective
def transform_plate_and_masks(plate, masks):
    w, h = CANVAS_SIZE
    plate_arr = np.array(plate)
    mask_arrs = [(ch, np.array(mask)) for ch, mask in masks]

    # Get original plate dimensions
    orig_w, orig_h = plate.size
    
    # random scale (scale the plate, not the canvas)
    scale = random.uniform(0.3, 0.8)
    scaled_w, scaled_h = int(orig_w * scale), int(orig_h * scale)
    
    # random rotation angle
    angle = random.uniform(-30, 30)
    
    # Calculate bounding box size needed to contain rotated plate
    # This ensures the full rotated plate is visible
    cos_a = abs(np.cos(np.radians(angle)))
    sin_a = abs(np.sin(np.radians(angle)))
    rotated_w = int(scaled_w * cos_a + scaled_h * sin_a)
    rotated_h = int(scaled_w * sin_a + scaled_h * cos_a)
    
    # Add padding to ensure full plate is visible
    padded_w = rotated_w + 40
    padded_h = rotated_h + 40
    
    # Create larger canvas for transformation
    transform_canvas_w = max(padded_w, scaled_w + 40)
    transform_canvas_h = max(padded_h, scaled_h + 40)
    
    # Resize plate first
    plate_scaled = plate.resize((scaled_w, scaled_h), Image.LANCZOS)
    
    # Create transformation canvas (larger than plate to avoid cropping)
    transform_canvas = np.zeros((transform_canvas_h, transform_canvas_w, 3), dtype=np.uint8)
    
    # Place scaled plate in center of transform canvas
    start_x = (transform_canvas_w - scaled_w) // 2
    start_y = (transform_canvas_h - scaled_h) // 2
    transform_canvas[start_y:start_y+scaled_h, start_x:start_x+scaled_w] = np.array(plate_scaled)
    
    # Apply rotation to the larger canvas
    M = cv2.getRotationMatrix2D((transform_canvas_w/2, transform_canvas_h/2), angle, 1)
    plate_transformed = cv2.warpAffine(transform_canvas, M, (transform_canvas_w, transform_canvas_h), 
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    # Find the actual content bounds (non-black pixels)
    gray = cv2.cvtColor(plate_transformed, cv2.COLOR_RGB2GRAY)
    coords = cv2.findNonZero(gray)
    if coords is not None:
        x, y, crop_w, crop_h = cv2.boundingRect(coords)
        plate_cropped = plate_transformed[y:y+crop_h, x:x+crop_w]
    else:
        plate_cropped = plate_transformed
    
    # Random position on final canvas
    final_h, final_w = plate_cropped.shape[:2]
    if final_w < w and final_h < h:
        tx = random.randint(0, w - final_w)
        ty = random.randint(0, h - final_h)
    else:
        tx = ty = 0
    
    # Place on background
    canvas_img = load_random_background()
    canvas_arr = np.array(canvas_img)
    
    # Create mask for blending (non-black pixels)
    mask = cv2.cvtColor(plate_cropped, cv2.COLOR_RGB2GRAY) > 0
    
    # Place the transformed plate
    end_y = min(ty + final_h, h)
    end_x = min(tx + final_w, w)
    actual_th = end_y - ty
    actual_tw = end_x - tx
    
    plate_section = plate_cropped[:actual_th, :actual_tw]
    mask_section = mask[:actual_th, :actual_tw]
    
    canvas_arr[ty:end_y, tx:end_x][mask_section] = plate_section[mask_section]

    # Transform masks similarly
    transformed_masks = []
    for ch, orig_mask in mask_arrs:
        # Resize mask
        mask_scaled = cv2.resize(orig_mask, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
        
        # Create mask transform canvas
        mask_transform_canvas = np.zeros((transform_canvas_h, transform_canvas_w), dtype=np.uint8)
        mask_transform_canvas[start_y:start_y+scaled_h, start_x:start_x+scaled_w] = mask_scaled
        
        # Apply same rotation
        mask_transformed = cv2.warpAffine(mask_transform_canvas, M, (transform_canvas_w, transform_canvas_h), 
                                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Crop to same bounds as plate
        if coords is not None:
            mask_cropped = mask_transformed[y:y+crop_h, x:x+crop_w]
        else:
            mask_cropped = mask_transformed
        
        # Place on full canvas
        full_mask = np.zeros((h, w), dtype=np.uint8)
        mask_crop_section = mask_cropped[:actual_th, :actual_tw]
        full_mask[ty:end_y, tx:end_x] = mask_crop_section
        
        transformed_masks.append((ch, full_mask))

    return canvas_arr, transformed_masks

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
        try:
            plate, masks = generate_random_plate()
            canvas, t_masks = transform_plate_and_masks(plate, masks)

            cv2.imshow('Augmented Plate', cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(0)
            if key != 32: break

            img_name = f'aug_{idx:05d}.jpg'
            cv2.imwrite(os.path.join(IMG_OUT, img_name), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            ann = generate_yolo_annotations(t_masks, canvas.shape)
            with open(os.path.join(LBL_OUT, img_name.replace('.jpg','.txt')), 'w') as f:
                f.write('\n'.join(ann))
            idx += 1
            print(f"Saved {img_name}")
            
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            break

    cv2.destroyAllWindows()