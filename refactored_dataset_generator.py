import os
import random
import numpy as np
import cv2
from PIL import Image
from refactored_plate_generator import generate_random_plate

# Configuration constants
BACKGROUND_DIR = 'backgrounds'
OUTPUT_DIR = 'augmented_dataset'
IMG_OUT = os.path.join(OUTPUT_DIR, 'images')
LBL_OUT = os.path.join(OUTPUT_DIR, 'labels')

CANVAS_SIZE = (512, 512)
SCALE_RANGE = (0.20, 1) 
ROTATION_RANGE = (-45, 45)
PERSPECTIVE_STRENGTH = 0.30
SHEAR_RANGE = (-0.15, 0.15) 
BLUR_RANGE = (0, 1) 
NOISE_INTENSITY = 0.0 
PERSPECTIVE_PROB = 0.25 
SHEAR_PROB = 0.25  
BLUR_PROB = 0.25 
NOISE_PROB = 0.0
PADDING_FACTOR = 3.0

# New motion blur and downscale/upscale parameters
MOTION_BLUR_PROB = 0.3
MOTION_BLUR_LENGTH_RANGE = (5, 15)  # Length of motion blur kernel
MOTION_BLUR_ANGLE_RANGE = (0, 360)  # Angle of motion blur

DOWNSCALE_UPSCALE_PROB = 0.4
DOWNSCALE_FACTOR_RANGE = (0.3, 0.7)  # Scale down factor

BASIC_AUGMENTATION_PROB = 0.3  # 30% chance of basic augmentation only

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(LBL_OUT, exist_ok=True)

def load_random_background():
    files = [f for f in os.listdir(BACKGROUND_DIR) if f.lower().endswith(('jpg','png','jpeg'))]
    bg = Image.open(os.path.join(BACKGROUND_DIR, random.choice(files))).convert('RGB')
    return bg.resize(CANVAS_SIZE, Image.LANCZOS)

def get_perspective_matrix(w, h, strength):
    dx = random.uniform(-strength, strength) * w
    dy = random.uniform(-strength, strength) * h
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([[dx, dy], [w - dx, dy], [w - dx, h - dy], [dx, h - dy]])
    return cv2.getPerspectiveTransform(src_pts, dst_pts)

def create_motion_blur_kernel(length, angle):
    """Create a motion blur kernel with specified length and angle"""
    # Create a kernel matrix
    kernel = np.zeros((length, length))
    
    # Calculate the center point
    center = length // 2
    
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Calculate dx and dy for the blur direction
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    
    # Create the motion blur line
    for i in range(length):
        x = int(center + (i - center) * dx)
        y = int(center + (i - center) * dy)
        
        # Ensure coordinates are within bounds
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1
    
    # Normalize the kernel
    kernel = kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel
    
    return kernel

def apply_motion_blur(img, use_basic_only=False):
    """Apply motion blur to the image"""
    if use_basic_only or random.random() >= MOTION_BLUR_PROB:
        return img
    
    # Generate random motion blur parameters
    length = random.randint(*MOTION_BLUR_LENGTH_RANGE)
    angle = random.uniform(*MOTION_BLUR_ANGLE_RANGE)
    
    # Create motion blur kernel
    kernel = create_motion_blur_kernel(length, angle)
    
    # Apply motion blur using filter2D
    blurred = cv2.filter2D(img, -1, kernel)
    
    return blurred

def apply_downscale_upscale(img, use_basic_only=False):
    """Apply downscale then smart upscale to the image"""
    if use_basic_only or random.random() >= DOWNSCALE_UPSCALE_PROB:
        return img
    
    original_shape = img.shape
    
    # Calculate downscale dimensions
    scale_factor = random.uniform(*DOWNSCALE_FACTOR_RANGE)
    new_h = max(1, int(original_shape[0] * scale_factor))
    new_w = max(1, int(original_shape[1] * scale_factor))
    
    # Downscale using area interpolation (good for downscaling)
    downscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Smart upscale using INTER_CUBIC or INTER_LANCZOS4
    upscale_method = random.choice([cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
    upscaled = cv2.resize(downscaled, (original_shape[1], original_shape[0]), interpolation=upscale_method)
    
    return upscaled

def add_blur(img, use_basic_only=False):
    if use_basic_only:
        return img  # No blur for basic augmentation
    if random.random() < BLUR_PROB:
        kernel_size = random.randint(1, BLUR_RANGE[1]) * 2 + 1
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return img

def add_noise(img, use_basic_only=False):
    # Noise completely removed
    return img

def apply_plate_effects(plate_img, use_basic_only=False):
    """Apply motion blur and downscale/upscale effects to plate only"""
    # Apply motion blur
    plate_img = apply_motion_blur(plate_img, use_basic_only)
    
    # Apply downscale/upscale
    plate_img = apply_downscale_upscale(plate_img, use_basic_only)
    
    return plate_img

def transform_plate_and_masks(plate, masks):
    w, h = CANVAS_SIZE
    mask_arrs = [(ch, np.array(mask)) for ch, mask in masks]
    orig_w, orig_h = plate.size

    # Determine if we should use basic augmentation only
    use_basic_only = random.random() < BASIC_AUGMENTATION_PROB

    scale = random.uniform(*SCALE_RANGE)
    scaled_w, scaled_h = int(orig_w * scale), int(orig_h * scale)
    transform_canvas_w = int(scaled_w * PADDING_FACTOR)
    transform_canvas_h = int(scaled_h * PADDING_FACTOR)
    start_x = (transform_canvas_w - scaled_w) // 2
    start_y = (transform_canvas_h - scaled_h) // 2

    plate_scaled = plate.resize((scaled_w, scaled_h), Image.LANCZOS)
    transform_canvas = np.zeros((transform_canvas_h, transform_canvas_w, 3), dtype=np.uint8)
    transform_canvas[start_y:start_y+scaled_h, start_x:start_x+scaled_w] = np.array(plate_scaled)
    
    mask_canvases = []
    for ch, orig_mask in mask_arrs:
        mask_scaled = cv2.resize(orig_mask, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
        mask_canvas = np.zeros((transform_canvas_h, transform_canvas_w), dtype=np.uint8)
        mask_canvas[start_y:start_y+scaled_h, start_x:start_x+scaled_w] = mask_scaled
        mask_canvases.append((ch, mask_canvas))
    
    # Always apply rotation
    angle = random.uniform(*ROTATION_RANGE)
    center = (transform_canvas_w/2, transform_canvas_h/2)
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply shear only if not using basic augmentation
    if not use_basic_only:
        shear_x = random.uniform(*SHEAR_RANGE) if random.random() < SHEAR_PROB else 0
        shear_y = random.uniform(*SHEAR_RANGE) if random.random() < SHEAR_PROB else 0
        S = np.array([[1, shear_x], [shear_y, 1]])
        R = M_rot[:, :2]
        M_rot[:, :2] = R @ S
    
    plate_warped = cv2.warpAffine(
        transform_canvas, M_rot, (transform_canvas_w, transform_canvas_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0)
    )
    
    masks_warped = []
    for ch, mask_canvas in mask_canvases:
        mask_warped = cv2.warpAffine(
            mask_canvas, M_rot, (transform_canvas_w, transform_canvas_h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        masks_warped.append((ch, mask_warped))
    
    # Apply perspective transformation only if not using basic augmentation
    if not use_basic_only and random.random() < PERSPECTIVE_PROB:
        M_perspective = get_perspective_matrix(transform_canvas_w, transform_canvas_h, PERSPECTIVE_STRENGTH)
        plate_warped = cv2.warpPerspective(
            plate_warped, M_perspective, (transform_canvas_w, transform_canvas_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0)
        )
        new_masks = []
        for ch, mask_warped in masks_warped:
            mask_persp = cv2.warpPerspective(
                mask_warped, M_perspective, (transform_canvas_w, transform_canvas_h),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            new_masks.append((ch, mask_persp))
        masks_warped = new_masks
    
    # Apply plate-specific effects (motion blur and downscale/upscale) before cropping
    plate_warped = apply_plate_effects(plate_warped, use_basic_only)
    
    gray = cv2.cvtColor(plate_warped, cv2.COLOR_RGB2GRAY)
    coords = cv2.findNonZero(gray)
    if coords is None:
        return None, None
        
    x, y, crop_w, crop_h = cv2.boundingRect(coords)
    plate_cropped = plate_warped[y:y+crop_h, x:x+crop_w]
    
    # Skip if plate is too large for canvas
    if plate_cropped.shape[0] > CANVAS_SIZE[1] or plate_cropped.shape[1] > CANVAS_SIZE[0]:
        return None, None
    
    masks_cropped = []
    for ch, mask_warped in masks_warped:
        masks_cropped.append((ch, mask_warped[y:y+crop_h, x:x+crop_w]))
    
    bg_img = load_random_background()
    canvas_arr = np.array(bg_img)
    final_h, final_w = plate_cropped.shape[:2]
    
    tx = random.randint(0, w - final_w) if final_w < w else 0
    ty = random.randint(0, h - final_h) if final_h < h else 0
    
    end_y = min(ty + final_h, h)
    end_x = min(tx + final_w, w)
    actual_th = end_y - ty
    actual_tw = end_x - tx
    
    plate_section = plate_cropped[:actual_th, :actual_tw]
    mask_gray = cv2.cvtColor(plate_section, cv2.COLOR_RGB2GRAY)
    _, alpha = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
    
    roi = canvas_arr[ty:end_y, tx:end_x]
    roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(alpha))
    roi_fg = cv2.bitwise_and(plate_section, plate_section, mask=alpha)
    canvas_arr[ty:end_y, tx:end_x] = cv2.add(roi_bg, roi_fg)
    
    final_masks = []
    for ch, mask_cropped in masks_cropped:
        mask_section = mask_cropped[:actual_th, :actual_tw]
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[ty:end_y, tx:end_x] = mask_section
        final_masks.append((ch, full_mask))
    
    # Apply blur and noise to entire image (background + plate)
    canvas_arr = add_blur(canvas_arr, use_basic_only)
    canvas_arr = add_noise(canvas_arr, use_basic_only)
    
    return canvas_arr, final_masks

def generate_yolo_segmentation_annotations(masks, image_shape):
    h, w = image_shape[:2]
    lines = []
    for ch, mask in masks:
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Skip very small contours
            if cv2.contourArea(contour) < 10:
                continue
                
            # Simplify the contour to reduce number of points
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convert to normalized coordinates
            points = []
            for point in approx:
                x, y = point[0]
                # Normalize coordinates (0-1 range)
                norm_x = x / w
                norm_y = y / h
                points.extend([norm_x, norm_y])
            
            # YOLO segmentation format: class_id x1 y1 x2 y2 x3 y3 ...
            # Use character mapping for class_id
            if len(points) >= 6:  # At least 3 points (triangle)
                points_str = ' '.join([f"{p:.6f}" for p in points])
                # Map character to class ID (you might want to adjust this mapping)
                char_to_class = {
                    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
                    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19,
                    'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25,
                    '0': 26, '1': 27, '2': 28, '3': 29, '4': 30, '5': 31, '6': 32, '7': 33, '8': 34, '9': 35
                }
                class_id = char_to_class.get(ch, 0)  # Default to 0 if character not found
                lines.append(f"{class_id} {points_str}")
    
    return lines

def overlay_masks(image, masks):
    display_img = image.copy()
    for i, (ch, mask) in enumerate(masks):
        hue = (i * 30) % 180
        color = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0,0]
        color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        
        y, x = np.where(mask > 0)
        display_img[y, x] = display_img[y, x] * 0.7 + np.array(color_rgb) * 0.3
    return display_img.astype(np.uint8)

if __name__ == '__main__':
    idx = 0
    while True:
        try:
            plate, masks = generate_random_plate()
            result = transform_plate_and_masks(plate, masks)
            if result is None:
                continue
                
            canvas, t_masks = result
            
            debug_img = overlay_masks(canvas.copy(), t_masks)
            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
            cv2.imshow('Augmented Plate', debug_img)
            
            key = cv2.waitKey(0)
            if key != 32:  # Space bar
                break

            img_name = f'aug_{idx:05d}.jpg'
            cv2.imwrite(os.path.join(IMG_OUT, img_name), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            ann = generate_yolo_segmentation_annotations(t_masks, canvas.shape)
            with open(os.path.join(LBL_OUT, img_name.replace('.jpg','.txt')), 'w') as f:
                f.write('\n'.join(ann))
            idx += 1
            print(f"Saved {img_name}")
            
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            break

    cv2.destroyAllWindows()