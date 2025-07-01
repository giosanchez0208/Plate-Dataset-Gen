import random
import numpy as np
from colorsys import rgb_to_hls, hls_to_rgb
from PIL import ImageDraw

def calculate_luminance(color):
    """Calculate the relative luminance of a color for contrast ratio calculation"""
    r, g, b = [x/255.0 for x in color]
    
    # Convert to linear RGB
    def to_linear(c):
        if c <= 0.03928:
            return c / 12.92
        else:
            return ((c + 0.055) / 1.055) ** 2.4
    
    r_linear = to_linear(r)
    g_linear = to_linear(g)
    b_linear = to_linear(b)
    
    # Calculate luminance
    return 0.2126 * r_linear + 0.7152 * g_linear + 0.0722 * b_linear

def contrast_ratio(color1, color2):
    """Calculate contrast ratio between two colors"""
    lum1 = calculate_luminance(color1)
    lum2 = calculate_luminance(color2)
    
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    
    return (lighter + 0.05) / (darker + 0.05)

def get_average_color_in_region(image, x, y, width, height):
    """Get average color in a specific region of the image"""
    # Ensure coordinates are within image bounds
    x1 = max(0, x - width//2)
    y1 = max(0, y - height//2)
    x2 = min(image.width, x + width//2)
    y2 = min(image.height, y + height//2)
    
    # Crop the region and calculate average color
    region = image.crop((x1, y1, x2, y2))
    region_array = np.array(region)
    avg_color = np.mean(region_array, axis=(0, 1))
    
    return tuple(map(int, avg_color))

def get_readable_color(background_color, min_contrast=5):
    """Generate a barely legible color based on background - light colors for dark backgrounds, dark colors for light backgrounds"""
    bg_luminance = calculate_luminance(background_color)
    
    # If background is dark, ONLY use light colors; if light, ONLY use dark colors
    if bg_luminance < 0.5:
        # Dark background - ONLY light/bright colors
        candidates = [
            (255, 255, 255),  # White
            (255, 255, 0),    # Yellow
            (0, 255, 255),    # Cyan
            (255, 0, 255),    # Magenta
            (255, 128, 0),    # Orange
            (128, 255, 128),  # Light green
            (255, 128, 255),  # Light pink
            (128, 255, 255),  # Light cyan
            (255, 255, 128),  # Light yellow
            (200, 200, 200),  # Light gray
            (255, 200, 200),  # Light red
            (200, 255, 200),  # Light green
            (200, 200, 255),  # Light blue
        ]
    else:
        # Light background - ONLY dark colors
        candidates = [
            (0, 0, 0),        # Black
            (128, 0, 0),      # Dark red
            (0, 128, 0),      # Dark green
            (0, 0, 128),      # Dark blue
            (128, 0, 128),    # Purple
            (128, 128, 0),    # Olive
            (0, 128, 128),    # Teal
            (128, 64, 0),     # Brown
            (64, 64, 64),     # Dark gray
            (100, 0, 100),    # Dark purple
            (0, 100, 100),    # Dark cyan
            (100, 100, 0),    # Dark yellow
            (80, 80, 80),     # Medium dark gray
        ]
    
    # Find colors that meet minimum contrast ratio
    good_colors = []
    for color in candidates:
        if contrast_ratio(background_color, color) >= min_contrast:
            good_colors.append(color)
    
    # If no good colors found, use guaranteed contrast colors
    if not good_colors:
        if bg_luminance < 0.5:
            return (255, 255, 255)  # White for dark backgrounds
        else:
            return (0, 0, 0)        # Black for light backgrounds
    
    return random.choice(good_colors)

def draw_text_with_outline(draw, position, text, font, fill_color, outline_color=(128, 128, 128), outline_width=1):
    """Draw text with subtle outline for better readability"""
    x, y = position
    
    # Draw outline
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color, anchor="mm")
    
    # Draw main text
    draw.text((x, y), text, font=font, fill=fill_color, anchor="mm")