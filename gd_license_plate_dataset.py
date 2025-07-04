import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import cv2
from gd_license_plate import generate_random_plate
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

# Configuration
DATASET_DIR = 'final_dataset'
IMAGES_DIR = os.path.join(DATASET_DIR, 'images')
LABELS_DIR = os.path.join(DATASET_DIR, 'labels')
BACKGROUNDS_DIR = 'backgrounds'
OUTPUT_SIZE = (512, 512)

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

# Load background files
background_files = []
if os.path.exists(BACKGROUNDS_DIR):
    background_files = [os.path.join(BACKGROUNDS_DIR, f) for f in os.listdir(BACKGROUNDS_DIR) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not background_files:
    raise RuntimeError(f"No background files found in {BACKGROUNDS_DIR}")

def augment_background(background_image):
    """Apply random hue shift and brightness/contrast to background"""
    # Convert to numpy array
    bg_array = np.array(background_image)
    
    # Random hue shift
    if random.random() < 0.7:
        # Convert to HSV
        hsv = cv2.cvtColor(bg_array, cv2.COLOR_RGB2HSV)
        hue_shift = random.randint(-30, 30)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + hue_shift) % 180
        hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)  
        bg_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Random brightness and contrast
    if random.random() < 0.8:
        brightness = random.uniform(0.7, 1.3)
        contrast = random.uniform(0.8, 1.2)
        
        bg_array = bg_array.astype(np.float32)
        bg_array = bg_array * contrast + (brightness - 1) * 128
        bg_array = np.clip(bg_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(bg_array)

def get_random_background():
    """Load and prepare a random background"""
    bg_path = random.choice(background_files)
    bg_image = Image.open(bg_path).convert('RGB')
    
    # Resize to fit output size while maintaining aspect ratio
    bg_image = bg_image.resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)
    
    # Apply augmentations
    bg_image = augment_background(bg_image)
    
    return bg_image

def apply_perspective_transform(image, annotations, max_perspective=0.3):
    """Apply perspective transformation to plate and update annotations"""
    if random.random() < 0.4:
        h, w = image.size[1], image.size[0]
        
        # Random perspective distortion
        perspective_strength = random.uniform(0.1, max_perspective)
        
        # Define source and destination points for perspective transform
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Random destination points
        dst_points = np.float32([
            [random.uniform(0, w * perspective_strength), random.uniform(0, h * perspective_strength)],
            [w - random.uniform(0, w * perspective_strength), random.uniform(0, h * perspective_strength)],
            [w - random.uniform(0, w * perspective_strength), h - random.uniform(0, h * perspective_strength)],
            [random.uniform(0, w * perspective_strength), h - random.uniform(0, h * perspective_strength)]
        ])
        
        # Get transformation matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply to image
        image_array = np.array(image)
        transformed = cv2.warpPerspective(image_array, matrix, (w, h))
        image = Image.fromarray(transformed)
        
        # Transform annotations
        for ann in annotations:
            if 'yolo_segments' in ann:
                new_segments = []
                for segment in ann['yolo_segments']:
                    new_segment = []
                    for i in range(0, len(segment), 2):
                        if i + 1 < len(segment):
                            x, y = segment[i] * w, segment[i + 1] * h
                            point = np.array([[[x, y]]], dtype=np.float32)
                            transformed_point = cv2.perspectiveTransform(point, matrix)
                            new_x = transformed_point[0][0][0] / w
                            new_y = transformed_point[0][0][1] / h
                            new_segment.extend([new_x, new_y])
                    new_segments.append(new_segment)
                ann['yolo_segments'] = new_segments
                
                # Update YOLO format
                segment_str = ' '.join([f"{coord:.6f}" for segment in new_segments for coord in segment])
                ann['yolo_format'] = f"0 {segment_str}"
    
    return image, annotations

def apply_shear_transform(image, annotations, max_shear=0.2):
    """Apply shear transformation to plate and update annotations"""
    if random.random() < 0.3:
        h, w = image.size[1], image.size[0]
        
        # Random shear
        shear_x = random.uniform(-max_shear, max_shear)
        shear_y = random.uniform(-max_shear, max_shear)
        
        # Shear matrix
        shear_matrix = np.array([
            [1, shear_x, 0],
            [shear_y, 1, 0]
        ], dtype=np.float32)
        
        # Apply to image
        image_array = np.array(image)
        transformed = cv2.warpAffine(image_array, shear_matrix, (w, h))
        image = Image.fromarray(transformed)
        
        # Transform annotations
        for ann in annotations:
            if 'yolo_segments' in ann:
                new_segments = []
                for segment in ann['yolo_segments']:
                    new_segment = []
                    for i in range(0, len(segment), 2):
                        if i + 1 < len(segment):
                            x, y = segment[i] * w, segment[i + 1] * h
                            point = np.array([[x, y, 1]], dtype=np.float32)
                            transformed_point = shear_matrix @ point.T
                            new_x = transformed_point[0][0] / w
                            new_y = transformed_point[1][0] / h
                            new_segment.extend([new_x, new_y])
                    new_segments.append(new_segment)
                ann['yolo_segments'] = new_segments
                
                # Update YOLO format
                segment_str = ' '.join([f"{coord:.6f}" for segment in new_segments for coord in segment])
                ann['yolo_format'] = f"0 {segment_str}"
    
    return image, annotations

def apply_rotation(image, annotations, max_rotation=30):
    """Apply rotation to plate and update annotations"""
    if random.random() < 0.5:
        angle = random.uniform(-max_rotation, max_rotation)
        h, w = image.size[1], image.size[0]
        
        # Rotation matrix
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply to image
        image_array = np.array(image)
        rotated = cv2.warpAffine(image_array, rotation_matrix, (w, h))
        image = Image.fromarray(rotated)
        
        # Transform annotations
        for ann in annotations:
            if 'yolo_segments' in ann:
                new_segments = []
                for segment in ann['yolo_segments']:
                    new_segment = []
                    for i in range(0, len(segment), 2):
                        if i + 1 < len(segment):
                            x, y = segment[i] * w, segment[i + 1] * h
                            point = np.array([[x, y, 1]], dtype=np.float32)
                            transformed_point = rotation_matrix @ point.T
                            new_x = transformed_point[0][0] / w
                            new_y = transformed_point[1][0] / h
                            new_segment.extend([new_x, new_y])
                    new_segments.append(new_segment)
                ann['yolo_segments'] = new_segments
                
                # Update YOLO format
                segment_str = ' '.join([f"{coord:.6f}" for segment in new_segments for coord in segment])
                ann['yolo_format'] = f"0 {segment_str}"
    
    return image, annotations

def apply_blur_effects(image):
    """Apply motion blur and gaussian blur"""
    # Motion blur
    if random.random() < 0.3:
        # Create motion blur kernel
        kernel_size = random.randint(5, 15)
        angle = random.uniform(0, 360)
        kernel = np.zeros((kernel_size, kernel_size))
        
        # Create line kernel for motion blur
        center = kernel_size // 2
        for i in range(kernel_size):
            x = int(center + (i - center) * math.cos(math.radians(angle)))
            y = int(center + (i - center) * math.sin(math.radians(angle)))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
        
        kernel = kernel / np.sum(kernel)
        
        # Apply motion blur
        image_array = np.array(image)
        blurred = cv2.filter2D(image_array, -1, kernel)
        image = Image.fromarray(blurred)
    
    # Gaussian blur
    elif random.random() < 0.4:
        blur_radius = random.uniform(0.5, 2.0)
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return image

def apply_pixellation(image):
    """Simulate upscaling artifacts (no model used)"""
    if random.random() < 0.2:
        w, h = image.size
        scale_factor = random.uniform(0.1, 0.5)
        small_size = (int(w * scale_factor), int(h * scale_factor))
        
        # Downscale with BICUBIC to simulate optical blur
        image_small = image.resize(small_size, Image.Resampling.BICUBIC)

        # Upscale back using BILINEAR or BICUBIC (creates smoother but still artifacted results)
        image = image_small.resize((w, h), Image.Resampling.BILINEAR)

    return image

def add_shine_effect(image):
    """Add shine effect to license plate"""
    if random.random() < 0.3:
        # Create shine overlay
        w, h = image.size
        shine = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        shine_draw = ImageDraw.Draw(shine)
        
        # Random shine parameters
        shine_x = random.randint(0, w)
        shine_y = random.randint(0, h)
        shine_width = random.randint(w // 4, w // 2)
        shine_height = random.randint(h // 8, h // 4)
        
        # Create elliptical shine
        shine_draw.ellipse([
            shine_x - shine_width // 2,
            shine_y - shine_height // 2,
            shine_x + shine_width // 2,
            shine_y + shine_height // 2
        ], fill=(255, 255, 255, random.randint(30, 80)))
        
        # Apply shine with blend mode
        image = image.convert('RGBA')
        image = Image.alpha_composite(image, shine)
        image = image.convert('RGB')
    
    return image

def transform_annotations_for_placement(annotations, plate_bbox, final_size):
    """Transform annotations from plate coordinates to final image coordinates"""
    plate_x, plate_y, plate_w, plate_h = plate_bbox
    final_w, final_h = final_size
    
    for ann in annotations:
        if 'yolo_segments' in ann:
            new_segments = []
            for segment in ann['yolo_segments']:
                new_segment = []
                for i in range(0, len(segment), 2):
                    if i + 1 < len(segment):
                        # Convert from plate-relative to absolute coordinates
                        x_plate = segment[i]
                        y_plate = segment[i + 1]
                        
                        # Transform to final image coordinates
                        x_final = (plate_x + x_plate * plate_w) / final_w
                        y_final = (plate_y + y_plate * plate_h) / final_h
                        
                        new_segment.extend([x_final, y_final])
                new_segments.append(new_segment)
            
            ann['yolo_segments'] = new_segments
            
            # Update YOLO format
            segment_str = ' '.join([f"{coord:.6f}" for segment in new_segments for coord in segment])
            ann['yolo_format'] = f"0 {segment_str}"
    
    return annotations

def create_dataset_sample(sample_id):
    """Create a single dataset sample with background and augmented license plate"""
    
    # Get random background
    background = get_random_background()
    
    # Generate license plate
    plate_image, plate_text, aug_level, annotations = generate_random_plate()
    
    # Apply only non-geometric augmentations to the plate first
    plate_image = apply_blur_effects(plate_image)
    plate_image = apply_pixellation(plate_image)
    plate_image = add_shine_effect(plate_image)
    
    # Random plate size (VERY IMPORTANT for scale variation)
    min_size = 40  # Even smaller plates for distant cars
    max_size = 350  # Slightly larger plates for close cars
    plate_size = random.randint(min_size, max_size)
    
    # Maintain aspect ratio
    original_w, original_h = plate_image.size
    aspect_ratio = original_w / original_h
    
    if random.random() < 0.5:
        # Width-based scaling
        new_w = plate_size
        new_h = int(plate_size / aspect_ratio)
    else:
        # Height-based scaling
        new_h = plate_size
        new_w = int(plate_size * aspect_ratio)
    
    plate_image = plate_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # NOW apply geometric transformations to the resized plate
    # Convert to RGBA if not already
    if plate_image.mode != 'RGBA':
        plate_image = plate_image.convert('RGBA')

    plate_array = np.array(plate_image)

    # Apply rotation to plate (50% chance for normal data)
    if random.random() < 0.5:
        angle = random.uniform(-25, 25)
        
        # Calculate new dimensions to fit the rotated image
        cos_angle = abs(np.cos(np.radians(angle)))
        sin_angle = abs(np.sin(np.radians(angle)))
        new_width = int(new_w * cos_angle + new_h * sin_angle)
        new_height = int(new_w * sin_angle + new_h * cos_angle)
        
        # Create a larger transparent canvas
        canvas = np.zeros((new_height, new_width, 4), dtype=np.uint8)
        
        # Calculate position to center the original plate
        start_x = (new_width - new_w) // 2
        start_y = (new_height - new_h) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = plate_array
        
        # Apply rotation to the canvas
        center = (new_width // 2, new_height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_canvas = cv2.warpAffine(canvas, rotation_matrix, (new_width, new_height))
        
        # Find the bounding box of non-transparent pixels
        alpha = rotated_canvas[:, :, 3]
        coords = np.column_stack(np.where(alpha > 0))
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Crop to the bounding box
            plate_array = rotated_canvas[y_min:y_max+1, x_min:x_max+1]
            
            # Update dimensions
            new_h, new_w = plate_array.shape[:2]
            
            # Transform annotations for rotation
            # We need to account for the canvas expansion, rotation, and then cropping
            for ann in annotations:
                if 'yolo_segments' in ann:
                    new_segments = []
                    for segment in ann['yolo_segments']:
                        new_segment = []
                        for i in range(0, len(segment), 2):
                            if i + 1 < len(segment):
                                # Original coordinates in the original plate
                                x_orig = segment[i] * (new_w if 'original_new_w' not in locals() else original_new_w)
                                y_orig = segment[i + 1] * (new_h if 'original_new_h' not in locals() else original_new_h)
                                
                                # Adjust for canvas expansion
                                x_canvas = x_orig + start_x
                                y_canvas = y_orig + start_y
                                
                                # Apply rotation
                                point = np.array([[x_canvas, y_canvas, 1]], dtype=np.float32)
                                transformed_point = rotation_matrix @ point.T
                                x_rot = transformed_point[0][0]
                                y_rot = transformed_point[1][0]
                                
                                # Adjust for cropping
                                x_final = x_rot - x_min
                                y_final = y_rot - y_min
                                
                                # Normalize to new dimensions
                                new_x = x_final / new_w
                                new_y = y_final / new_h
                                
                                new_segment.extend([new_x, new_y])
                        new_segments.append(new_segment)
                    ann['yolo_segments'] = new_segments
            
            # Store original dimensions for perspective transform
            original_new_w, original_new_h = new_w, new_h

    # Apply perspective transform to plate (30% chance, more intense)
    if random.random() < 0.3:
        # Milder perspective distortion for realistic driving angles
        perspective_strength = random.uniform(0.05, 0.15)
        
        src_points = np.float32([[0, 0], [new_w, 0], [new_w, new_h], [0, new_h]])
        
        # Create realistic perspective for different viewing angles
        perspective_type = random.choice(['left_receding', 'right_receding', 'left_approaching', 'right_approaching', 'top_down', 'bottom_up'])
        
        if perspective_type == 'left_receding':  # Car moving away, viewed from left
            dst_points = np.float32([
                [0, random.uniform(0, new_h * 0.05)],
                [new_w - random.uniform(new_w * perspective_strength, new_w * perspective_strength * 2), random.uniform(0, new_h * 0.05)],
                [new_w - random.uniform(new_w * perspective_strength, new_w * perspective_strength * 2), new_h - random.uniform(0, new_h * 0.05)],
                [0, new_h - random.uniform(0, new_h * 0.05)]
            ])
        elif perspective_type == 'right_receding':  # Car moving away, viewed from right
            dst_points = np.float32([
                [random.uniform(new_w * perspective_strength, new_w * perspective_strength * 2), random.uniform(0, new_h * 0.05)],
                [new_w, random.uniform(0, new_h * 0.05)],
                [new_w, new_h - random.uniform(0, new_h * 0.05)],
                [random.uniform(new_w * perspective_strength, new_w * perspective_strength * 2), new_h - random.uniform(0, new_h * 0.05)]
            ])
        elif perspective_type == 'left_approaching':  # Car approaching, viewed from left
            dst_points = np.float32([
                [random.uniform(new_w * perspective_strength, new_w * perspective_strength * 2), random.uniform(0, new_h * 0.05)],
                [new_w, random.uniform(0, new_h * 0.05)],
                [new_w, new_h - random.uniform(0, new_h * 0.05)],
                [random.uniform(new_w * perspective_strength, new_w * perspective_strength * 2), new_h - random.uniform(0, new_h * 0.05)]
            ])
        elif perspective_type == 'right_approaching':  # Car approaching, viewed from right
            dst_points = np.float32([
                [0, random.uniform(0, new_h * 0.05)],
                [new_w - random.uniform(new_w * perspective_strength, new_w * perspective_strength * 2), random.uniform(0, new_h * 0.05)],
                [new_w - random.uniform(new_w * perspective_strength, new_w * perspective_strength * 2), new_h - random.uniform(0, new_h * 0.05)],
                [0, new_h - random.uniform(0, new_h * 0.05)]
            ])
        elif perspective_type == 'top_down':  # Slight top-down view
            dst_points = np.float32([
                [random.uniform(0, new_w * perspective_strength), 0],
                [new_w - random.uniform(0, new_w * perspective_strength), 0],
                [new_w - random.uniform(0, new_w * perspective_strength * 0.5), new_h],
                [random.uniform(0, new_w * perspective_strength * 0.5), new_h]
            ])
        else:  # bottom_up - Slight bottom-up view
            dst_points = np.float32([
                [random.uniform(0, new_w * perspective_strength * 0.5), 0],
                [new_w - random.uniform(0, new_w * perspective_strength * 0.5), 0],
                [new_w - random.uniform(0, new_w * perspective_strength), new_h],
                [random.uniform(0, new_w * perspective_strength), new_h]
            ])
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        plate_array = cv2.warpPerspective(plate_array, matrix, (new_w, new_h))
        
        # Transform annotations for perspective
        for ann in annotations:
            if 'yolo_segments' in ann:
                new_segments = []
                for segment in ann['yolo_segments']:
                    new_segment = []
                    for i in range(0, len(segment), 2):
                        if i + 1 < len(segment):
                            # Convert from normalized plate coordinates to absolute coordinates
                            x, y = segment[i] * new_w, segment[i + 1] * new_h
                            point = np.array([[[x, y]]], dtype=np.float32)
                            transformed_point = cv2.perspectiveTransform(point, matrix)
                            # Convert back to normalized plate coordinates
                            new_x = transformed_point[0][0][0] / new_w
                            new_y = transformed_point[0][0][1] / new_h
                            new_segment.extend([new_x, new_y])
                    new_segments.append(new_segment)
                ann['yolo_segments'] = new_segments

    # Convert back to PIL Image, preserving transparency
    plate_image = Image.fromarray(plate_array, 'RGBA')
    
    # Random position on background
    bg_w, bg_h = background.size
    max_x = bg_w - new_w
    max_y = bg_h - new_h
    
    # Initialize position variables
    pos_x = 0
    pos_y = 0
    
    if max_x > 0 and max_y > 0:
        pos_x = random.randint(0, max_x)
        pos_y = random.randint(0, max_y)
        
        # Place transformed plate on background
        if plate_image.mode == 'RGBA':
            background.paste(plate_image, (pos_x, pos_y), plate_image)
        else:
            background.paste(plate_image, (pos_x, pos_y))
        
        # Transform annotations for final placement and scale
        for ann in annotations:
            if 'yolo_segments' in ann:
                new_segments = []
                for segment in ann['yolo_segments']:
                    new_segment = []
                    for i in range(0, len(segment), 2):
                        if i + 1 < len(segment):
                            # Scale from plate coordinates to image coordinates
                            x_plate = segment[i] * new_w
                            y_plate = segment[i + 1] * new_h
                            
                            # Translate to final position
                            x_final = (pos_x + x_plate) / bg_w
                            y_final = (pos_y + y_plate) / bg_h
                            
                            # Clamp to valid range [0, 1]
                            x_final = max(0.0, min(1.0, x_final))
                            y_final = max(0.0, min(1.0, y_final))
                            
                            new_segment.extend([x_final, y_final])
                    new_segments.append(new_segment)
                ann['yolo_segments'] = new_segments
                
                # Update YOLO format
                segment_str = ' '.join([f"{coord:.6f}" for segment in new_segments for coord in segment])
                ann['yolo_format'] = f"0 {segment_str}"
        
        # Save image
        image_name = f'sample_{sample_id:06d}.jpg'
        image_path = os.path.join(IMAGES_DIR, image_name)
        background.save(image_path, quality=90)
        
        # Save annotations
        label_path = os.path.join(LABELS_DIR, image_name.replace('.jpg', '.txt'))
        with open(label_path, 'w') as f:
            yolo_lines = [ann['yolo_format'] for ann in annotations if 'yolo_format' in ann]
            f.write('\n'.join(yolo_lines))
        
        return {
            'image_path': image_path,
            'label_path': label_path,
            'plate_text': plate_text,
            'plate_size': (new_w, new_h),
            'plate_position': (pos_x, pos_y),
            'augmentation_level': aug_level,
            'num_characters': len(annotations)
        }
    else:
        # Plate too big for background, skip this sample
        return None
    
def create_dataset(num_samples=1000):
    """Create the complete dataset"""
    print(f"Creating dataset with {num_samples} samples...")
    print(f"Output directory: {DATASET_DIR}")
    print(f"Image size: {OUTPUT_SIZE}")
    print(f"Found {len(background_files)} background images")
    
    successful_samples = 0
    
    for i in range(num_samples):
        try:
            result = create_dataset_sample(i)
            if result:
                successful_samples += 1
                
                if i % 100 == 0:
                    print(f"Generated {i} samples... Latest: '{result['plate_text']}' "
                          f"({result['plate_size'][0]}x{result['plate_size'][1]}) "
                          f"at ({result['plate_position'][0]}, {result['plate_position'][1]})")
        
        except Exception as e:
            print(f"Error creating sample {i}: {e}")
    
    print(f"\nDataset creation complete!")
    print(f"Successfully generated {successful_samples} samples")
    print(f"Images saved to: {IMAGES_DIR}")
    print(f"Labels saved to: {LABELS_DIR}")

def generate_samples_batch(start_id, end_id, progress_queue=None):
    """Generate a batch of samples"""
    results = []
    for sample_id in range(start_id, end_id):
        try:
            result = create_dataset_sample(sample_id)
            if result:
                results.append(result)
            
            # Report progress if queue provided
            if progress_queue:
                progress_queue.put(1)
                
        except Exception as e:
            print(f"Error generating sample {sample_id}: {e}")
            continue
    
    return results

def generate_dataset_multithreaded(total_samples, num_threads=4, batch_size=100):
    """Generate dataset using multiple threads"""
    
    # Create progress tracking
    progress_queue = queue.Queue()
    completed = 0
    
    def progress_tracker():
        nonlocal completed
        while True:
            try:
                progress_queue.get(timeout=1)
                completed += 1
                if completed % 100 == 0:
                    print(f"Progress: {completed}/{total_samples} samples generated")
            except queue.Empty:
                break
    
    # Start progress tracker thread
    progress_thread = threading.Thread(target=progress_tracker, daemon=True)
    progress_thread.start()
    
    # Create batches
    batches = []
    for i in range(0, total_samples, batch_size):
        end_id = min(i + batch_size, total_samples)
        batches.append((i, end_id))
    
    all_results = []
    
    # Process batches with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(generate_samples_batch, start_id, end_id, progress_queue): (start_id, end_id)
            for start_id, end_id in batches
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            start_id, end_id = future_to_batch[future]
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
                print(f"Completed batch {start_id}-{end_id}: {len(batch_results)} successful samples")
            except Exception as e:
                print(f"Error in batch {start_id}-{end_id}: {e}")
    
    # Wait for progress tracker to finish
    progress_thread.join(timeout=2)
    
    print(f"Dataset generation complete! Generated {len(all_results)} samples out of {total_samples} attempts")
    return all_results

if __name__ == '__main__':
    # Generate 40000 samples using 8 threads
    results = generate_dataset_multithreaded(
        total_samples=40000,
        num_threads=8,
        batch_size=50
    )