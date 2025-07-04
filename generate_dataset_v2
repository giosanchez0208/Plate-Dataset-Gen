import os
import random
time_import = __import__('time')
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
import numpy as np
import cv2

# Directories
OUTPUT_DIR = 'dataset'
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')
LABELS_DIR = os.path.join(OUTPUT_DIR, 'labels')
MASKS_DIR = os.path.join(OUTPUT_DIR, 'masks')
FONTS_DIR = 'fonts'
RIZAL_BG_PATH = os.path.join('assets', 'rizal.png')
OBSCURATIONS_DIR = os.path.join('assets', 'obscurations')
MARGIN = 5
OFFSET = 4

# Ensure dirs
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

# Config
COLOR_PAIRS = [
    ('#FFFFFF','#E35205'),('#FFFFFF','#000000'),('#FFFFFF',"#022B17"),('#FFFFFF','#F90000'),
    ('#FFFFFF','#062964'),('#F6C60D','#000000'),('#FFA500','#000000'),('#0A47AD','#000000')
]
FORMATS = ['LLL DDDD','LLL DDD','LL DDDD','DLL DDD','LL DDDDD']
ASPECT_RATIOS = {'car':2.79,'motorcycle':1.74}
font_files = [os.path.join(FONTS_DIR,f) for f in os.listdir(FONTS_DIR) if f.lower().endswith(('.ttf','.otf'))]
if not font_files: raise RuntimeError(f"No font files in {FONTS_DIR}")
obscuration_files = [os.path.join(OBSCURATIONS_DIR,f) for f in os.listdir(OBSCURATIONS_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))] if os.path.exists(OBSCURATIONS_DIR) else []

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
        tc = '#047d61'
    else:
        bg = Image.new('RGBA', (w,h), bgc)

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

    # Effects
    fg = white_stroke(fg, width=1)
    fg = drop_shadow(fg, offset=(2,2), blur=1, alpha=90)
    fg = deterioration(fg, chips=random.randint(5,10))
    fg = directional_emboss(fg, angle=random.choice([60,120,180]), depth=1.2)
    
    plate = Image.alpha_composite(bg, fg)
    plate = apply_natural_obscurations(plate)
    plate_rgb = plate.convert('RGB')

    name = f'plate_{id:05d}.jpg'
    plate_rgb.save(os.path.join(IMAGES_DIR, name))

    masks = []
    labels = []
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
        masks.append((text[i], mask))
        mask.save(os.path.join(MASKS_DIR, name.replace('.jpg', f'_char_{i}.png')))

        np_mask = np.array(mask)
        ys, xs = np.where(np_mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            cx = ((x0 + x1) / 2) / w
            cy = ((y0 + y1) / 2) / h
            cw = (x1 - x0) / w
            ch = (y1 - y0) / h
            labels.append(f"0 {cx:.6f} {cy:.6f} {cw:.6f} {ch:.6f}")

    with open(os.path.join(LABELS_DIR, name.replace('.jpg','.txt')), 'w') as f:
        f.write("\n".join(labels))

    return plate_rgb, masks

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
