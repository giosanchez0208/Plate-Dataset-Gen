import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np

# Configuration
OUTPUT_DIR = 'dataset'
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')
LABELS_DIR = os.path.join(OUTPUT_DIR, 'labels')
FONTS_DIR = 'fonts'
RIZAL_BG_PATH = os.path.join('assets', 'rizal.png')
OBSCURATIONS_DIR = os.path.join('assets', 'obscurations')
MARGIN = 5

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

COLOR_PAIRS = [
    ('#FFFFFF', '#E35205'),
    ('#FFFFFF', '#000000'),
    ('#FFFFFF', '#026434'),
    ('#FFFFFF', '#F90000'),
    ('#FFFFFF', '#062964'),
    ('#F6C60D', '#000000'),
    ('#FFA500', '#000000'),
    ('#0A47AD', '#000000')
]

FORMATS = ['LLL DDDD', 'LLL DDD', 'LL DDDD', 'DLL DDD', 'LL DDDDD']
ASPECT_RATIOS = {'car': 2.79, 'motorcycle': 1.74}

# Load font files
font_files = [os.path.join(FONTS_DIR, f) for f in os.listdir(FONTS_DIR) if f.lower().endswith(('.ttf', '.otf'))]
if not font_files:
    raise RuntimeError(f"No font files found in {FONTS_DIR}")

# Load obscuration files
obscuration_files = []
if os.path.exists(OBSCURATIONS_DIR):
    obscuration_files = [os.path.join(OBSCURATIONS_DIR, f) for f in os.listdir(OBSCURATIONS_DIR) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def generate_plate_text():
    pattern = random.choice(FORMATS)
    return ''.join(
        random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') if c == 'L' else
        random.choice('0123456789') if c == 'D' else c
        for c in pattern
    )

def create_varied_obscurations(image):
    """Create varied color, shape, and size obscurations"""
    draw = ImageDraw.Draw(image)
    num_obscurations = random.randint(1, 6)
    
    for _ in range(num_obscurations):
        # Varied size
        size = random.randint(3, 40)
        x = random.randint(0, image.width - size)
        y = random.randint(0, image.height - size)
        
        # Varied color (dirt, rust, wear colors)
        color_options = [
            (random.randint(40, 100), random.randint(30, 70), random.randint(20, 50)),  # Brown/dirt
            (random.randint(80, 120), random.randint(50, 90), random.randint(30, 60)),  # Rust
            (random.randint(60, 100), random.randint(60, 100), random.randint(60, 100)),  # Gray
            (random.randint(20, 60), random.randint(40, 80), random.randint(20, 50)),  # Dark green
        ]
        color = random.choice(color_options) + (random.randint(100, 200),)
        
        # Varied shapes
        shape_type = random.choice(['rectangle', 'ellipse', 'polygon'])
        
        if shape_type == 'rectangle':
            draw.rectangle([x, y, x + size, y + size], fill=color)
        elif shape_type == 'ellipse':
            draw.ellipse([x, y, x + size, y + size], fill=color)
        else:  # polygon
            # Create irregular polygon
            points = []
            for i in range(random.randint(3, 6)):
                px = x + random.randint(0, size)
                py = y + random.randint(0, size)
                points.append((px, py))
            draw.polygon(points, fill=color)
    
    return image

def apply_letter_deterioration(image):
    """Apply letter deterioration effects"""
    if random.random() < 0.3:
        # Create small holes/chips in letters
        draw = ImageDraw.Draw(image)
        for _ in range(random.randint(1, 3)):
            x = random.randint(0, image.width - 5)
            y = random.randint(0, image.height - 5)
            hole_size = random.randint(2, 8)
            draw.ellipse([x, y, x + hole_size, y + hole_size], fill=(0, 0, 0, 0))
    
    if random.random() < 0.2:
        # Add edge wear by eroding edges
        # Convert to numpy for edge detection
        img_array = np.array(image)
        alpha = img_array[:, :, 3]
        
        # Find edges
        edges = np.zeros_like(alpha)
        edges[1:-1, 1:-1] = (alpha[1:-1, 1:-1] > 128) & (
            (alpha[:-2, 1:-1] < 128) | (alpha[2:, 1:-1] < 128) |
            (alpha[1:-1, :-2] < 128) | (alpha[1:-1, 2:] < 128)
        )
        
        # Randomly remove some edge pixels
        wear_mask = np.random.random(edges.shape) < 0.3
        wear_locations = edges & wear_mask
        img_array[wear_locations] = [0, 0, 0, 0]
        
        image = Image.fromarray(img_array, 'RGBA')
    
    return image

def apply_emboss_to_letters_only(fg_image):
    """Apply emboss effect only to letters and border"""
    if random.random() < 0.2:
        # Create embossed version
        embossed = fg_image.filter(ImageFilter.EMBOSS)
        
        # Use multiply blend mode effect
        # Convert to numpy for custom blending
        fg_array = np.array(fg_image).astype(np.float32)
        embossed_array = np.array(embossed).astype(np.float32)
        
        # Multiply blend where there's content
        mask = fg_array[:, :, 3] > 0
        result = fg_array.copy()
        
        for c in range(3):  # RGB channels
            result[mask, c] = (fg_array[mask, c] * embossed_array[mask, c]) / 255.0
        
        return Image.fromarray(result.astype(np.uint8), 'RGBA')
    
    return fg_image

def apply_drop_shadow_to_letters_only(fg_image):
    """Apply drop shadow only to letters and border"""
    if random.random() < 0.2:
        # Create shadow layer
        shadow = Image.new('RGBA', fg_image.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        
        # Get the content area
        bbox = fg_image.getbbox()
        if bbox:
            # Create shadow offset
            offset_x = random.randint(1, 3)
            offset_y = random.randint(1, 3)
            shadow_alpha = random.randint(50, 100)
            
            # Paste the foreground as shadow with offset
            shadow_temp = Image.new('RGBA', fg_image.size, (0, 0, 0, 0))
            shadow_temp.paste(fg_image, (offset_x, offset_y))
            
            # Convert shadow to black with alpha
            shadow_array = np.array(shadow_temp)
            shadow_array[:, :, :3] = 0  # Make it black
            shadow_array[:, :, 3] = (shadow_array[:, :, 3] * shadow_alpha / 255).astype(np.uint8)
            
            shadow = Image.fromarray(shadow_array, 'RGBA')
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=1))
            
            # Composite shadow with original
            return Image.alpha_composite(shadow, fg_image)
    
    return fg_image

def apply_choke_and_spill(fg_image):
    """Apply choke (shrink) and spill (expand) effects"""
    if random.random() < 0.3:
        effect_type = random.choice(['choke', 'spill'])
        intensity = random.randint(1, 2)
        
        if effect_type == 'choke':
            # Shrink the letters (erode)
            fg_array = np.array(fg_image)
            alpha = fg_array[:, :, 3]
            
            # Simple erosion
            kernel_size = intensity * 2 + 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            # Manual erosion
            eroded = np.zeros_like(alpha)
            for i in range(intensity, alpha.shape[0] - intensity):
                for j in range(intensity, alpha.shape[1] - intensity):
                    window = alpha[i-intensity:i+intensity+1, j-intensity:j+intensity+1]
                    if np.all(window > 128):
                        eroded[i, j] = alpha[i, j]
            
            fg_array[:, :, 3] = eroded
            return Image.fromarray(fg_array, 'RGBA')
            
        else:  # spill
            # Expand the letters (dilate)
            fg_array = np.array(fg_image)
            alpha = fg_array[:, :, 3]
            
            # Simple dilation
            dilated = alpha.copy()
            for i in range(intensity, alpha.shape[0] - intensity):
                for j in range(intensity, alpha.shape[1] - intensity):
                    window = alpha[i-intensity:i+intensity+1, j-intensity:j+intensity+1]
                    if np.any(window > 128):
                        dilated[i, j] = 255
            
            fg_array[:, :, 3] = dilated
            return Image.fromarray(fg_array, 'RGBA')
    
    return fg_image

def add_white_stroke(fg_image):
    """Add white stroke around letters"""
    if random.random() < 0.3:
        stroke_width = random.randint(1, 5)
        
        # Create stroke layer
        stroke_image = Image.new('RGBA', fg_image.size, (0, 0, 0, 0))
        
        # Dilate the original to create stroke
        fg_array = np.array(fg_image)
        alpha = fg_array[:, :, 3]
        
        # Create stroke by dilation
        stroke_alpha = np.zeros_like(alpha)
        for i in range(stroke_width, alpha.shape[0] - stroke_width):
            for j in range(stroke_width, alpha.shape[1] - stroke_width):
                window = alpha[i-stroke_width:i+stroke_width+1, j-stroke_width:j+stroke_width+1]
                if np.any(window > 128):
                    stroke_alpha[i, j] = 255
        
        # Remove original letter area from stroke
        stroke_alpha[alpha > 128] = 0
        
        # Create white stroke
        stroke_array = np.zeros_like(fg_array)
        stroke_array[:, :, :3] = 255  # White
        stroke_array[:, :, 3] = stroke_alpha
        
        stroke_image = Image.fromarray(stroke_array, 'RGBA')
        
        # Composite stroke under original
        return Image.alpha_composite(stroke_image, fg_image)
    
    return fg_image

def apply_natural_obscurations(image):
    """Apply natural obscurations from PNG files"""
    if obscuration_files and random.random() < 0.4:
        obscuration_path = random.choice(obscuration_files)
        try:
            obscuration = Image.open(obscuration_path).convert('RGBA')
            
            # Resize obscuration to cover the whole plate
            obscuration = obscuration.resize(image.size, Image.Resampling.LANCZOS)
            
            # Random opacity
            opacity = random.uniform(0.2, 0.8)
            obscuration_array = np.array(obscuration)
            obscuration_array[:, :, 3] = (obscuration_array[:, :, 3] * opacity).astype(np.uint8)
            obscuration = Image.fromarray(obscuration_array, 'RGBA')
            
            # Composite over the plate
            return Image.alpha_composite(image.convert('RGBA'), obscuration).convert('RGB')
        except Exception as e:
            print(f"Error loading obscuration {obscuration_path}: {e}")
    
    return image

def apply_letter_augmentations(image):
    """Apply various augmentations to letters and border only"""
    # Apply deterioration
    image = apply_letter_deterioration(image)
    
    # Apply emboss effect (letters only)
    image = apply_emboss_to_letters_only(image)
    
    # Apply drop shadow (letters only)
    image = apply_drop_shadow_to_letters_only(image)
    
    # Apply choke and spill
    image = apply_choke_and_spill(image)
    
    # Add white stroke
    image = add_white_stroke(image)
    
    # Apply varied obscurations
    image = create_varied_obscurations(image)
    
    return image

def create_plate(id, plate_type='car', augmentation_level='random'):
    """
    Create a license plate with various augmentation levels
    
    augmentation_level: 'none', 'light', 'medium', 'heavy', 'random'
    """
    width = 350
    height = int(width / ASPECT_RATIOS[plate_type])
    BORDER_WIDTH = max(1, int(height * 0.02))

    bg_color, txt_color = random.choice(COLOR_PAIRS)
    use_rizal = random.random() < 0.1 and os.path.exists(RIZAL_BG_PATH)
    bg = Image.open(RIZAL_BG_PATH).convert('RGBA').resize((width, height)) if use_rizal else Image.new('RGBA', (width, height), bg_color)
    text_color = '#047d61' if use_rizal else txt_color

    text = generate_plate_text()
    font_path = random.choice(font_files)
    font_size = height
    draw_base = ImageDraw.Draw(bg)
    
    # Find optimal font size
    while font_size > 1:
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw_base.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if text_w + 2 * (MARGIN + BORDER_WIDTH) <= width and text_h + 2 * (MARGIN + BORDER_WIDTH) <= height:
            break
        font_size -= 1

    spacing = random.randint(-2, 3)
    x = (width - text_w) // 2
    y = (height - text_h) // 2
    V_PAD = max(1, int(text_h * 0.1))

    # Create foreground layer with text and border
    fg = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw_fg = ImageDraw.Draw(fg)
    draw_fg.rectangle([
        x - BORDER_WIDTH, y - BORDER_WIDTH - V_PAD,
        x + text_w + BORDER_WIDTH, y + text_h + BORDER_WIDTH + V_PAD
    ], outline=text_color, width=BORDER_WIDTH)
    draw_fg.text((x, y), text, font=font, fill=text_color, spacing=spacing)

    # Determine augmentation level
    if augmentation_level == 'random':
        aug_level = random.choices(['none', 'light', 'medium', 'heavy'], weights=[0.2, 0.3, 0.3, 0.2])[0]
    else:
        aug_level = augmentation_level

    # Apply augmentations based on level
    if aug_level != 'none':
        # Adjust probability of each augmentation based on level
        if aug_level == 'light':
            # Reduce chances for light augmentation
            original_random = random.random
            random.random = lambda: original_random() * 0.5
        elif aug_level == 'heavy':
            # Increase chances for heavy augmentation
            original_random = random.random
            random.random = lambda: original_random() * 1.5
        
        fg = apply_letter_augmentations(fg)
        
        # Restore original random function
        if aug_level in ['light', 'heavy']:
            random.random = original_random

    # Composite layers
    final = Image.alpha_composite(bg, fg)
    
    # Apply natural obscurations (affects whole image)
    if aug_level in ['medium', 'heavy']:
        final = apply_natural_obscurations(final)
    
    final = final.convert('RGB')
    
    # Save image
    image_name = f'plate_{id:05d}.jpg'
    final_path = os.path.join(IMAGES_DIR, image_name)
    final.save(final_path)

    # Generate YOLO-format annotations and build annotations list
    labels = []
    annotations = []
    offset = x
    for ch in text:
        ch_bbox = draw_fg.textbbox((offset, y), ch, font=font)
        cx = (ch_bbox[0] + ch_bbox[2]) / 2 / width
        cy = (ch_bbox[1] + ch_bbox[3]) / 2 / height
        cw = (ch_bbox[2] - ch_bbox[0]) / width
        ch_h = (ch_bbox[3] - ch_bbox[1]) / height
        line = f"0 {cx:.6f} {cy:.6f} {cw:.6f} {ch_h:.6f}"
        labels.append(line)

        # create corresponding annotation dict
        x0 = cx - cw/2; y0 = cy - ch_h/2
        x1 = cx + cw/2; y1 = cy - ch_h/2
        x2 = cx + cw/2; y2 = cy + ch_h/2
        x3 = cx - cw/2; y3 = cy + ch_h/2
        annotations.append({
            'bbox': (cx, cy, cw, ch_h),
            'yolo_segments': [[x0, y0, x1, y1, x2, y2, x3, y3]],
            'yolo_format': line
        })
        offset += (ch_bbox[2] - ch_bbox[0]) + spacing

    # Save labels
    label_path = os.path.join(LABELS_DIR, image_name.replace('.jpg', '.txt'))
    with open(label_path, 'w') as f:
        f.write('\n'.join(labels))
    
    return final, text, aug_level, annotations

def generate_random_plate():
    """Generate a single random license plate with random augmentations"""
    import time
    plate_id = int(time.time() * 1000) % 100000  # Use timestamp for unique ID
    # Now returns annotations as well
    return create_plate(plate_id, augmentation_level='random')

if __name__ == '__main__':
    # Generate dataset with varied augmentation levels
    total_plates = 1000
    
    for i in range(total_plates):
        plate_image, plate_text, aug_level = create_plate(i, augmentation_level='random')
        
        if i % 100 == 0:
            print(f"Generated {i} plates... Latest: {plate_text} (augmentation: {aug_level})")
    
    print(f"Generated {total_plates} plates in {IMAGES_DIR} and labels in {LABELS_DIR}.")
    
    # Example of generating a single random plate
    print("\nGenerating example random plate...")
    example_plate, example_text, example_aug = generate_random_plate()
    print(f"Example plate: {example_text} with {example_aug} augmentation")