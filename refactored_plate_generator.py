import os
import random
time_import = __import__('time')
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
import numpy as np
import cv2

# Directories (only needed for assets)
FONTS_DIR = 'fonts'
RIZAL_BG_PATH = os.path.join('assets', 'rizal.png')
OBSCURATIONS_DIR = os.path.join('assets', 'obscurations')
OBSCURATIONS_SMALL_DIR = os.path.join('assets', 'obscurations_small') 
MARGIN = 5
OFFSET = 4

# Config
COLOR_PAIRS = [
    ('#FFFFFF','#E35205'),('#FFFFFF','#000000'),('#FFFFFF',"#022B17"),('#FFFFFF','#F90000'),
    ('#FFFFFF','#062964'),('#F6C60D','#000000'),('#FFA500','#000000'),('#0A47AD','#000000')
]
FORMATS = ['LLL DDDD','LLL DDD','LL DDDD','DLL DDD','LL DDDDD', 'DDDD DDDDDDD']
ASPECT_RATIOS = {'car':2.79,'motorcycle':1.74}
font_files = [os.path.join(FONTS_DIR,f) for f in os.listdir(FONTS_DIR) if f.lower().endswith(('.ttf','.otf'))]
if not font_files: raise RuntimeError(f"No font files in {FONTS_DIR}")
obscuration_files = [os.path.join(OBSCURATIONS_DIR,f) for f in os.listdir(OBSCURATIONS_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))] if os.path.exists(OBSCURATIONS_DIR) else []
obscuration_files = [os.path.join(OBSCURATIONS_DIR,f) for f in os.listdir(OBSCURATIONS_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))] if os.path.exists(OBSCURATIONS_DIR) else []
obscuration_small_files = [os.path.join(OBSCURATIONS_SMALL_DIR,f) for f in os.listdir(OBSCURATIONS_SMALL_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))] if os.path.exists(OBSCURATIONS_SMALL_DIR) else []
# Effects

def directional_emboss(fg, angle=135, depth=1.0):
    kernel = [ -1, -1, 0, -1,  8*depth, -1, 0, -1, -1 ]
    emb = fg.filter(ImageFilter.Kernel((3,3),kernel,scale=None,offset=128))
    return ImageChops.blend(fg, emb, alpha=0.6)

def drop_shadow(fg, offset=(2,2), blur=2, alpha=80):
    w,h = fg.size
    tmp = Image.new('RGBA',(w,h),(0,0,0,0))
    tmp.paste(fg,offset)
    arr = np.array(tmp)
    arr[:,:,:3]=0
    arr[:,:,3] = (arr[:,:,3]*alpha/255).astype(np.uint8)
    shadow = Image.fromarray(arr,'RGBA').filter(ImageFilter.GaussianBlur(blur))
    return Image.alpha_composite(shadow,fg)

def white_stroke(fg, width=1):
    arr = np.array(fg)
    a = arr[:,:,3]
    stroke = np.zeros_like(arr)
    for r in range(width,a.shape[0]-width):
        for c in range(width,a.shape[1]-width):
            if np.any(a[r-width:r+width+1,c-width:c+width+1]>128): stroke[r,c,3]=255
    stroke[arr[:,:,3]>128,3]=0
    stroke[:,:,:3]=255
    return Image.alpha_composite(Image.fromarray(stroke,'RGBA'),fg)

def deterioration(fg, chips=3):
    draw = ImageDraw.Draw(fg)
    for _ in range(chips):
        x,y = random.randint(0,fg.width-5),random.randint(0,fg.height-5)
        r = random.randint(5, 15)
        draw.ellipse((x,y,x+r,y+r),fill=(0,0,0,0))
    return fg

def apply_natural_obscurations(img):
    if obscuration_files and random.random() < 0.4:
        for _ in range(random.randint(1, 2)):
            try:
                obs = Image.open(random.choice(obscuration_files)).convert('RGBA').resize(img.size)
                alpha = obs.split()[3].point(lambda p: int(p * random.uniform(0.2, 0.7)))
                new_obs = obs.copy()
                new_obs.putalpha(alpha)
                img = Image.alpha_composite(img.convert('RGBA'), new_obs)
            except Exception:
                continue
        return img
    return img

# Plate generation

def generate_plate_text():
    return ''.join(
        random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') if c=='L' else
        random.choice('0123456789') if c=='D' else c
        for c in random.choice(FORMATS)
    )

def create_plate(id):
    plate_type = 'car' if random.random() < 0.8 else 'motorcycle'
    w = 350
    h = int(w / ASPECT_RATIOS[plate_type])
    bw = max(1, int(h * 0.02)) if plate_type == 'car' else max(1, int(h * 0.03))

    bgc, tc = random.choice(COLOR_PAIRS)
    if random.random() < 0.1 and os.path.exists(RIZAL_BG_PATH):
        bg = Image.open(RIZAL_BG_PATH).convert('RGBA').resize((w,h))
        tc = "#012C23"
        plate_bg_color = '#F0F0F0'  # Off-white for Rizal background
    else:
        bg = Image.new('RGBA', (w,h), bgc)
        plate_bg_color = bgc

    text = generate_plate_text()
    fs = h
    font = ImageFont.truetype(random.choice(font_files), fs)
    d_bg = ImageDraw.Draw(bg)

    while True:
        font = ImageFont.truetype(font.path, fs)
        bx = d_bg.textbbox((0,0), text, font=font)
        tw, th = bx[2]-bx[0], bx[3]-bx[1]
        if tw + 2*(MARGIN + bw) <= w and th + 2*(MARGIN + bw) <= h:
            break
        fs -= 1

    spacing = random.randint(-2, 3)
    x_start = (w - tw) // 2
    y = (h - th) // 2
    vpad = max(1, int(th * 0.1))

    fg = Image.new('RGBA', (w,h))
    d_fg = ImageDraw.Draw(fg)
    d_fg.rectangle([x_start-bw, y-bw-vpad, x_start+tw+bw, y+th+bw+vpad], outline=tc, width=bw)
    d_fg.text((x_start, y-OFFSET), text, font=font, fill=tc, spacing=spacing)

    char_masks = []
    for i in range(len(text)):
        if text[i] == ' ': 
            continue
            
        full = Image.new('L', (w, h), 0)
        part = Image.new('L', (w, h), 0)
        d_full = ImageDraw.Draw(full)
        d_part = ImageDraw.Draw(part)
        d_full.text((x_start, y-OFFSET), text[:i+1], font=font, fill=255, spacing=spacing)
        d_part.text((x_start, y-OFFSET), text[:i], font=font, fill=255, spacing=spacing)
        diff = ImageChops.subtract(full, part)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(np.array(diff), kernel, iterations=1)
        mask = Image.fromarray(dilated)
        
        opacity = random.uniform(0.7, 1.0)
        fg_array = np.array(fg)
        mask_array = np.array(mask)
        char_pixels = mask_array > 0
        fg_array[char_pixels, 3] = (fg_array[char_pixels, 3] * opacity).astype(np.uint8)
        fg = Image.fromarray(fg_array, 'RGBA')
        
        char_masks.append((text[i], mask))

    # Apply small obscurations to individual letters
    if obscuration_small_files and random.random() < 0.7:
        for idx, (char, mask) in enumerate(char_masks):
            if char == ' ' or random.random() > 0.3:
                continue
                
            overlay_path = random.choice(obscuration_small_files)
            try:
                bbox = mask.getbbox()
                if not bbox: continue
                
                overlay = Image.open(overlay_path).convert('RGBA')
                w_bb, h_bb = bbox[2]-bbox[0], bbox[3]-bbox[1]
                
                # Scale to letter height
                scale_factor = h_bb / overlay.height
                new_size = (int(overlay.width * scale_factor), h_bb)
                overlay = overlay.resize(new_size, Image.LANCZOS)
                
                # Use subtraction blend mode for black obscurations
                overlay_arr = np.array(overlay)
                alpha = overlay_arr[:, :, 3]
                
                # Create subtraction effect - where overlay is opaque, subtract from fg
                subtraction_layer = np.zeros_like(overlay_arr)
                subtraction_layer[:, :, 3] = alpha
                subtraction_overlay = Image.fromarray(subtraction_layer, 'RGBA')
                
                pos = (
                    bbox[0] + (w_bb - new_size[0]) // 2,
                    bbox[1] + (h_bb - new_size[1]) // 2
                )
                
                # Create positioned overlay
                positioned_overlay = Image.new('RGBA', fg.size, (0,0,0,0))
                positioned_overlay.paste(subtraction_overlay, pos)
                
                # Apply subtraction blend - darken the fg where overlay alpha is present
                fg_arr = np.array(fg)
                pos_arr = np.array(positioned_overlay)
                
                # Where overlay has alpha, reduce the fg alpha (creating holes/darkening)
                alpha_mask = pos_arr[:, :, 3] > 0
                fg_arr[alpha_mask, 3] = (fg_arr[alpha_mask, 3] * (1 - pos_arr[alpha_mask, 3] / 255.0)).astype(np.uint8)
                
                fg = Image.fromarray(fg_arr, 'RGBA')
                
            except Exception as e:
                continue

    fg = white_stroke(fg, width=1)
    fg = drop_shadow(fg, offset=(2,2), blur=1, alpha=90)
    fg = deterioration(fg, chips=random.randint(5,10))
    fg = directional_emboss(fg, angle=random.choice([60,120,180]), depth=1.2)
    
    plate = Image.alpha_composite(bg, fg)
    plate = apply_natural_obscurations(plate)
    plate_rgb = plate.convert('RGB')

    return plate_rgb, char_masks

def generate_random_plate():
    return create_plate(int(time_import.time()*1000) % 100000)

# Interactive viewer

def main():
    while True:
        plate, masks = generate_random_plate()
        cv2.imshow('Plate', cv2.cvtColor(np.array(plate), cv2.COLOR_RGB2BGR))
        if cv2.waitKey(0) == 32:
            over = np.array(plate).copy()
            for ch, mask in masks:
                print(ch)
                m = np.array(mask) > 0
                over[m] = [0,0,255]
            cv2.imshow('Masks Overlay', cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(0) != 32: break
        else:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()