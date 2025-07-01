import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import math

class LicensePlateAugmentations:
    def __init__(self):
        # Augmentation probabilities - tune these based on your needs
        self.probabilities = {
            'motion_blur': 0.3,
            'normal_blur': 0.25,
            'partial_block': 0.2,
            'pixelate': 0.15,
            'shine': 0.25,
            'shadow': 0.3,
            'emboss': 0.2,
            'deterioration': 0.25,
            'rust': 0.2,
            'perspective': 0.3,
            'rotation': 0.4,
            'shear': 0.3,
            'erase_area': 0.15,
            'outline': 0.35,
            'transparency': 0.2,
            'color_jitter_small': 0.4,
            'color_jitter_large': 0.15,
            'scale_horizontal': 0.35,
            'scale_vertical': 0.3,
            'elastic_transform': 0.25,
            'chromatic_aberration': 0.2
        }
    
    def should_apply(self, augmentation_name):
        """Check if augmentation should be applied based on probability"""
        return random.random() < self.probabilities[augmentation_name]
    
    def motion_blur(self, image, intensity=None):
        """Apply motion blur to simulate camera movement"""
        if not self.should_apply('motion_blur'):
            return image
            
        if intensity is None:
            intensity = random.randint(5, 15)
        
        # Create motion blur kernel
        angle = random.uniform(0, 180)
        kernel = np.zeros((intensity, intensity))
        kernel[int((intensity-1)/2), :] = np.ones(intensity)
        kernel = kernel / intensity
        
        # Rotate kernel
        M = cv2.getRotationMatrix2D((intensity/2, intensity/2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (intensity, intensity))
        
        return cv2.filter2D(image, -1, kernel)
    
    def normal_blur(self, image, intensity=None):
        """Apply Gaussian blur"""
        if not self.should_apply('normal_blur'):
            return image
            
        if intensity is None:
            intensity = random.randint(1, 5)
        
        kernel_size = intensity * 2 + 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def partial_block(self, image, num_blocks=None):
        """Partially block text with random rectangles (dirt, scratches, etc.)"""
        if not self.should_apply('partial_block'):
            return image
            
        if num_blocks is None:
            num_blocks = random.randint(1, 3)
        
        h, w = image.shape[:2]
        result = image.copy()
        
        for _ in range(num_blocks):
            # Random block size and position
            block_w = random.randint(w//20, w//8)
            block_h = random.randint(h//20, h//8)
            x = random.randint(0, max(1, w - block_w))
            y = random.randint(0, max(1, h - block_h))
            
            # Random block color (dirt-like colors)
            colors = [(50, 40, 30), (80, 70, 60), (40, 35, 25), (60, 50, 40)]
            color = random.choice(colors)
            
            cv2.rectangle(result, (x, y), (x + block_w, y + block_h), color, -1)
            
            # Add some transparency
            alpha = random.uniform(0.3, 0.8)
            result[y:y+block_h, x:x+block_w] = cv2.addWeighted(
                image[y:y+block_h, x:x+block_w], 1-alpha,
                result[y:y+block_h, x:x+block_w], alpha, 0
            )
        
        return result
    
    def scale_horizontal(self, image, scale_factor=None):
        """Apply horizontal scaling (stretch/compress)"""
        if not self.should_apply('scale_horizontal'):
            return image
            
        if scale_factor is None:
            scale_factor = random.uniform(0.7, 1.4)  # 0.7 = compress, 1.4 = stretch
        
        h, w = image.shape[:2]
        new_w = int(w * scale_factor)
        
        # Resize with new width
        scaled = cv2.resize(image, (new_w, h), interpolation=cv2.INTER_LINEAR)
        
        # If scaled image is larger, crop to original size
        if new_w > w:
            start_x = (new_w - w) // 2
            return scaled[:, start_x:start_x + w]
        # If scaled image is smaller, pad to original size
        elif new_w < w:
            result = np.full((h, w, 3), 255, dtype=np.uint8)  # White padding
            start_x = (w - new_w) // 2
            result[:, start_x:start_x + new_w] = scaled
            return result
        else:
            return scaled
    
    def scale_vertical(self, image, scale_factor=None):
        """Apply vertical scaling (stretch/compress)"""
        if not self.should_apply('scale_vertical'):
            return image
            
        if scale_factor is None:
            scale_factor = random.uniform(0.8, 1.3)  # Less extreme than horizontal
        
        h, w = image.shape[:2]
        new_h = int(h * scale_factor)
        
        # Resize with new height
        scaled = cv2.resize(image, (w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # If scaled image is larger, crop to original size
        if new_h > h:
            start_y = (new_h - h) // 2
            return scaled[start_y:start_y + h, :]
        # If scaled image is smaller, pad to original size
        elif new_h < h:
            result = np.full((h, w, 3), 255, dtype=np.uint8)  # White padding
            start_y = (h - new_h) // 2
            result[start_y:start_y + new_h, :] = scaled
            return result
        else:
            return scaled
    
    def elastic_transform(self, image, alpha=None, sigma=None):
        """Apply elastic deformation (like text printed on flexible material)"""
        if not self.should_apply('elastic_transform'):
            return image
            
        if alpha is None:
            alpha = random.uniform(30, 60)  # Strength of deformation
        if sigma is None:
            sigma = random.uniform(4, 8)   # Smoothness of deformation
        
        h, w = image.shape[:2]
        
        # Generate random displacement fields
        dx = np.random.uniform(-1, 1, (h, w)).astype(np.float32)
        dy = np.random.uniform(-1, 1, (h, w)).astype(np.float32)
        
        # Smooth the displacement fields
        dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha
        
        # Create coordinate grids
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Apply displacement
        x_displaced = (x + dx).astype(np.float32)
        y_displaced = (y + dy).astype(np.float32)
        
        # Remap the image
        return cv2.remap(image, x_displaced, y_displaced, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    def chromatic_aberration(self, image, intensity=None):
        """Apply chromatic aberration (color channel misalignment)"""
        if not self.should_apply('chromatic_aberration'):
            return image
            
        if intensity is None:
            intensity = random.randint(1, 3)  # Pixel offset
        
        h, w = image.shape[:2]
        
        # Split channels
        b, g, r = cv2.split(image)
        
        # Create displacement matrices for each channel
        # Red channel - shift right/up
        M_r = np.float32([[1, 0, intensity], [0, 1, -intensity]])
        r_shifted = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Blue channel - shift left/down  
        M_b = np.float32([[1, 0, -intensity], [0, 1, intensity]])
        b_shifted = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Green channel stays in place
        g_shifted = g
        
        # Merge channels back
        return cv2.merge([b_shifted, g_shifted, r_shifted])
    
    def pixelate(self, image, factor=None):
        """Apply pixelation effect"""
        if not self.should_apply('pixelate'):
            return image
            
        if factor is None:
            factor = random.randint(2, 6)
        
        h, w = image.shape[:2]
        # Resize down and back up
        small = cv2.resize(image, (w//factor, h//factor), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    def add_shine(self, image, intensity=None):
        """Add shine/glare effect on top of text"""
        if not self.should_apply('shine'):
            return image
            
        if intensity is None:
            intensity = random.uniform(0.3, 0.7)
        
        h, w = image.shape[:2]
        
        # Create shine mask
        shine_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Random shine area
        num_shines = random.randint(1, 2)
        for _ in range(num_shines):
            center_x = random.randint(w//4, 3*w//4)
            center_y = random.randint(h//4, 3*h//4)
            radius = random.randint(min(w, h)//8, min(w, h)//4)
            
            cv2.circle(shine_mask, (center_x, center_y), radius, 255, -1)
        
        # Apply Gaussian blur to make it more natural
        shine_mask = cv2.GaussianBlur(shine_mask, (21, 21), 0)
        
        # Convert to 3-channel and normalize
        shine_mask = cv2.cvtColor(shine_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        
        # Apply shine
        result = image.astype(np.float32)
        shine_color = np.array([255, 255, 200], dtype=np.float32)  # Warm white
        
        result = result + shine_mask * shine_color * intensity
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def add_shadow(self, image, intensity=None):
        """Add shadow effect on top of text"""
        if not self.should_apply('shadow'):
            return image
            
        if intensity is None:
            intensity = random.uniform(0.2, 0.6)
        
        h, w = image.shape[:2]
        
        # Create shadow mask
        shadow_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Random shadow areas
        num_shadows = random.randint(1, 3)
        for _ in range(num_shadows):
            # Create irregular shadow shapes
            points = []
            center_x = random.randint(0, w)
            center_y = random.randint(0, h)
            
            for i in range(6):
                angle = i * 60 + random.randint(-30, 30)
                radius = random.randint(min(w, h)//8, min(w, h)//3)
                x = int(center_x + radius * math.cos(math.radians(angle)))
                y = int(center_y + radius * math.sin(math.radians(angle)))
                points.append([x, y])
            
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(shadow_mask, [points], 255)
        
        # Blur shadow
        shadow_mask = cv2.GaussianBlur(shadow_mask, (15, 15), 0)
        shadow_mask = shadow_mask.astype(np.float32) / 255.0
        
        # Apply shadow
        result = image.astype(np.float32)
        result = result * (1 - shadow_mask[..., np.newaxis] * intensity)
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def emboss(self, image, strength=None):
        """Apply emboss effect"""
        if not self.should_apply('emboss'):
            return image
            
        if strength is None:
            strength = random.uniform(0.5, 1.5)
        
        # Emboss kernel
        kernel = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]]) * strength
        
        embossed = cv2.filter2D(image, -1, kernel)
        # Add some of original image back
        alpha = random.uniform(0.3, 0.7)
        return cv2.addWeighted(image, 1-alpha, embossed, alpha, 128)
    
    def deterioration(self, image, intensity=None):
        """Apply deterioration/aging effect"""
        if not self.should_apply('deterioration'):
            return image
            
        if intensity is None:
            intensity = random.uniform(0.1, 0.4)
        
        h, w = image.shape[:2]
        
        # Create noise
        noise = np.random.normal(0, 25, (h, w, 3)).astype(np.float32)
        
        # Create deterioration pattern
        deterioration_mask = np.random.random((h, w)).astype(np.float32)
        deterioration_mask = cv2.GaussianBlur(deterioration_mask, (5, 5), 0)
        deterioration_mask = (deterioration_mask > 0.7).astype(np.float32)
        
        result = image.astype(np.float32)
        result = result + noise * intensity
        result = result - deterioration_mask[..., np.newaxis] * 30 * intensity
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def add_rust(self, image, intensity=None):
        """Add rust/corrosion effect"""
        if not self.should_apply('rust'):
            return image
            
        if intensity is None:
            intensity = random.uniform(0.2, 0.5)
        
        h, w = image.shape[:2]
        
        # Create rust pattern
        rust_mask = np.random.random((h, w)).astype(np.float32)
        rust_mask = cv2.GaussianBlur(rust_mask, (3, 3), 0)
        rust_mask = (rust_mask > 0.6).astype(np.float32)
        
        # Rust colors (reddish-brown)
        rust_colors = np.array([30, 60, 120], dtype=np.float32)  # BGR format
        
        result = image.astype(np.float32)
        rust_effect = rust_mask[..., np.newaxis] * rust_colors * intensity
        result = cv2.addWeighted(result.astype(np.uint8), 1, rust_effect.astype(np.uint8), 1, 0)
        
        return result
    
    def perspective_transform(self, image, intensity=None):
        """Apply perspective transformation"""
        if not self.should_apply('perspective'):
            return image
            
        if intensity is None:
            intensity = random.uniform(0.15, 0.35)  # Increased from 0.05-0.15
        
        h, w = image.shape[:2]
        
        # Define source points (corners of the image)
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Define destination points with random distortion
        dst_points = np.float32([
            [random.uniform(-w*intensity, w*intensity), 
             random.uniform(-h*intensity, h*intensity)],
            [w + random.uniform(-w*intensity, w*intensity), 
             random.uniform(-h*intensity, h*intensity)],
            [w + random.uniform(-w*intensity, w*intensity), 
             h + random.uniform(-h*intensity, h*intensity)],
            [random.uniform(-w*intensity, w*intensity), 
             h + random.uniform(-h*intensity, h*intensity)]
        ])
        
        # Apply perspective transformation
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    def rotate(self, image, angle=None):
        """Apply rotation"""
        if not self.should_apply('rotation'):
            return image
            
        if angle is None:
            angle = random.uniform(-5, 5)  # Reduced from -15 to 15
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    def shear(self, image, intensity=None):
        """Apply shear transformation"""
        if not self.should_apply('shear'):
            return image
            
        if intensity is None:
            intensity = random.uniform(-0.4, 0.4)  # Increased from -0.2 to 0.2
        
        h, w = image.shape[:2]
        
        # Shear matrix
        shear_matrix = np.float32([[1, intensity, 0], [0, 1, 0]])
        return cv2.warpAffine(image, shear_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    def erase_random_area(self, image, num_areas=None):
        """Erase random rectangular areas"""
        if not self.should_apply('erase_area'):
            return image
            
        if num_areas is None:
            num_areas = random.randint(1, 2)
        
        h, w = image.shape[:2]
        result = image.copy()
        
        for _ in range(num_areas):
            # Random erased area size and position
            area_w = random.randint(w//15, w//6)
            area_h = random.randint(h//15, h//6)
            x = random.randint(0, max(1, w - area_w))
            y = random.randint(0, max(1, h - area_h))
            
            # Fill with background-like color
            bg_color = tuple(map(int, np.mean(image, axis=(0, 1))))
            cv2.rectangle(result, (x, y), (x + area_w, y + area_h), bg_color, -1)
        
        return result
    
    def add_outline(self, image, thickness=None):
        """Add outline effect to text areas"""
        if not self.should_apply('outline'):
            return image
            
        if thickness is None:
            thickness = random.randint(1, 3)
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges
        kernel = np.ones((thickness, thickness), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Convert edges back to 3-channel
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Choose outline color (usually darker)
        outline_colors = [(0, 0, 0), (50, 50, 50), (100, 100, 100)]
        outline_color = random.choice(outline_colors)
        
        # Apply outline
        result = image.copy()
        mask = edges > 0
        result[mask] = outline_color
        
        return result
    
    def adjust_transparency(self, image, alpha=None):
        """Adjust image transparency (simulate faded text)"""
        if not self.should_apply('transparency'):
            return image
            
        if alpha is None:
            alpha = random.uniform(0.6, 0.9)
        
        # Create a background (assuming white background for license plates)
        background = np.full_like(image, 255)
        
        return cv2.addWeighted(image, alpha, background, 1-alpha, 0)
    
    def color_jitter_small(self, image):
        """Apply small color jitter"""
        if not self.should_apply('color_jitter_small'):
            return image
        
        # Small adjustments
        brightness = random.uniform(0.9, 1.1)
        contrast = random.uniform(0.9, 1.1)
        
        result = cv2.convertScaleAbs(image, alpha=contrast, beta=(brightness-1)*50)
        
        # Slight hue shift
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hue_shift = random.randint(-10, 10)
        hsv[:, :, 0] = cv2.add(hsv[:, :, 0], hue_shift)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return result
    
    def color_jitter_large(self, image):
        """Apply large color jitter"""
        if not self.should_apply('color_jitter_large'):
            return image
        
        # Large adjustments
        brightness = random.uniform(0.6, 1.4)
        contrast = random.uniform(0.7, 1.3)
        saturation = random.uniform(0.7, 1.3)
        
        result = cv2.convertScaleAbs(image, alpha=contrast, beta=(brightness-1)*50)
        
        # Significant hue and saturation changes
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hue_shift = random.randint(-30, 30)
        hsv[:, :, 0] = cv2.add(hsv[:, :, 0], hue_shift)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], saturation)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return result
    
    def apply_all_augmentations(self, image):
        """Apply all augmentations with their respective probabilities"""
        result = image.copy()
        
        # Apply augmentations in order (some should be applied before others)
        
        # Geometric transformations first
        result = self.perspective_transform(result)
        result = self.rotate(result)
        result = self.shear(result)
        result = self.scale_horizontal(result)
        result = self.scale_vertical(result)
        result = self.elastic_transform(result)
        
        # Blur effects
        result = self.motion_blur(result)
        result = self.normal_blur(result)
        result = self.pixelate(result)
        
        # Color and lighting effects
        result = self.color_jitter_small(result)
        result = self.color_jitter_large(result)
        result = self.chromatic_aberration(result)
        result = self.adjust_transparency(result)
        
        # Surface effects
        result = self.emboss(result)
        result = self.deterioration(result)
        result = self.add_rust(result)
        
        # Overlay effects (should be applied on top)
        result = self.add_shadow(result)
        result = self.add_shine(result)
        result = self.add_outline(result)
        
        # Occlusion effects (should be last)
        result = self.partial_block(result)
        result = self.erase_random_area(result)
        
        return result


# Usage example:
def augment_license_plate_image(image_path):
    """Example function showing how to use the augmentations"""
    # Load image
    image = cv2.imread(image_path)
    
    # Create augmentation instance
    augmentor = LicensePlateAugmentations()
    
    # Apply all augmentations
    augmented = augmentor.apply_all_augmentations(image)
    
    return augmented


# Integration with your existing code
def integrate_augmentations_to_main(image_cv):
    """Function to integrate with your main.py"""
    augmentor = LicensePlateAugmentations()
    return augmentor.apply_all_augmentations(image_cv)


# Modified version of your process_final_image function
def process_final_image_with_augmentations(img):
    """Enhanced version of your process_final_image function"""
    augmentor = LicensePlateAugmentations()
    variations = []
    
    # Create multiple augmented versions
    for i in range(27):  # Generate 27 variations like your original
        # Apply augmentations first
        augmented = augmentor.apply_all_augmentations(img)
        
        # Then apply your brightness/contrast/saturation variations
        brightness = random.choice([0.5, 1.0, 1.5])
        contrast = random.choice([0.5, 1.0, 1.5])
        saturation = random.choice([0.5, 1.0, 1.5])
        
        variation = cv2.convertScaleAbs(augmented, alpha=contrast, beta=int((brightness - 1) * 50))
        
        hsv = cv2.cvtColor(variation, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], saturation)
        variation = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        variations.append(variation)
    
    # Save variations
    for i, var in enumerate(variations):
        filename = f"outputs/output_variation_{i+1}.jpg"
        cv2.imwrite(filename, var)
    
    return variations