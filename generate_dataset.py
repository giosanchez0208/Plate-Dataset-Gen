import os
import random
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from color_smart import get_average_color_in_region, get_readable_color
from augmentations import LicensePlateAugmentations

# initializations
characters = [chr(c) for c in range(ord('A'), ord('Z') + 1)] + [str(d) for d in range(10)]
WIDTH, HEIGHT = 512, 512
background_files = [f for f in os.listdir("backgrounds")
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif'))]
font_files = [f for f in os.listdir("fonts")
              if f.lower().endswith(('.ttf', '.otf'))]

def get_random_char():
    return random.choice(characters)

def process_final_image(img):
    variations = []
    
    for brightness in [0.5, 1.0, 1.5]:
        for contrast in [0.5, 1.0, 1.5]:
            for saturation in [0.5, 1.0, 1.5]:
                variation = img.copy()
                
                variation = cv2.convertScaleAbs(variation, alpha=contrast, beta=int((brightness - 1) * 50))
                
                hsv = cv2.cvtColor(variation, cv2.COLOR_BGR2HSV)
                hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], saturation)
                variation = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
                variations.append(variation)
    
    for i, var in enumerate(variations):
        filename = f"outputs/output_variation_{i+1}.jpg"
        cv2.imwrite(filename, var)

def main():
    
    """ STEP 1: INITIALIZE AN IMAGE """
    img = Image.new('RGB', (512, 512), color='white')

    """ STEP 2: SET A RANDOM IMAGE FROM BACKGROUNDS FOLDER AS BACKGROUND """
    # choose random background
    random_file = random.choice(background_files)
    # set random background as image
    background_image = Image.open(os.path.join("backgrounds", random_file)).convert("RGB").resize((WIDTH, HEIGHT))
    image_cv = cv2.cvtColor(np.array(background_image), cv2.COLOR_RGB2BGR)
    
    """ STEP 3. CHOOSE RANDOM LETTER AND APPLY RANDOM AUGMENTATIONS """
    num_letters = random.randint(3, 7)
    augmented_letters = []
    
    # Convert to PIL for background color analysis
    pil_background = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    
    # Create augmentor instance
    augmentor = LicensePlateAugmentations()

    for _ in range(num_letters):
        char = get_random_char()
        font_size = random.randint(15, 60)
        rotation = random.randint(-45, 45)
        
        # Choose position first
        x = random.randint(100, WIDTH-100)
        y = random.randint(100, HEIGHT-100)
        
        # Get background color at text position and generate readable color
        background_color = get_average_color_in_region(pil_background, x, y, font_size, font_size)
        color = get_readable_color(background_color)
        
        font = ImageFont.truetype(os.path.join("fonts", random.choice(font_files)), font_size)
        
        # Create individual letter image with more padding for augmentations
        letter_size = max(200, font_size * 4)  # Increased padding
        temp_img = Image.new('RGBA', (letter_size, letter_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)
        draw.text((letter_size//2, letter_size//2), char, font=font, fill=color, anchor="mm")
        
        # Convert PIL to OpenCV format for augmentations
        letter_array = np.array(temp_img)
        letter_bgr = cv2.cvtColor(letter_array, cv2.COLOR_RGBA2BGR)
        
        # Apply augmentations to individual letter
        augmented_letter_bgr = augmentor.apply_all_augmentations(letter_bgr)
        
        # Convert back to PIL with alpha channel
        augmented_letter_rgb = cv2.cvtColor(augmented_letter_bgr, cv2.COLOR_BGR2RGB)
        augmented_letter_pil = Image.fromarray(augmented_letter_rgb)
        
        # Create alpha mask from original letter to maintain transparency
        alpha_mask = temp_img.split()[-1]  # Get alpha channel from original
        
        # Apply the alpha mask to augmented letter
        augmented_letter_pil.putalpha(alpha_mask)
        
        # Apply rotation to the augmented letter
        rotated = augmented_letter_pil.rotate(rotation, expand=True)
        
        augmented_letters.append((rotated, x, y))


    """ STEP 4. ADD AUGMENTED LETTERS TO IMAGE"""
    pil_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    
    for letter, x, y in augmented_letters:
        # Adjust position to center the rotated letter
        paste_x = x - letter.width // 2
        paste_y = y - letter.height // 2
        
        # Ensure the letter fits within image bounds
        paste_x = max(0, min(paste_x, WIDTH - letter.width))
        paste_y = max(0, min(paste_y, HEIGHT - letter.height))
        
        pil_image.paste(letter, (paste_x, paste_y), letter)
    
    image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    """ STEP X: PROCESS IMAGE WITH MULTIPLE VARIATIONS """
    # process final created image
    process_final_image(image_cv)
    

if __name__ == "__main__":
    main()