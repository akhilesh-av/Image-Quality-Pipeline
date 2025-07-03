import os
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import imutils
import piexif
from skimage import exposure, filters, measure, img_as_float
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from io import BytesIO
import json
from datetime import datetime

class ImageQualityPipeline:
    def __init__(self, image_path, output_dir='output'):
        self.image_path = image_path
        self.output_dir = output_dir
        self.report = {
            'filename': os.path.basename(image_path),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'steps': {},
            'quality_score': 0,
            'passed': True,
            'flags': []
        }
        
        os.makedirs(output_dir, exist_ok=True)
        self.image = None
        self.processed_image = None
        self.grayscale = False
        
    def run_pipeline(self):
        try:
            # Step 1: File Validation
            self.step_1_file_validation()
            
            # Step 2: Corruption Check
            self.step_2_corruption_check()
            
            # Step 3: Color Mode Normalization
            self.step_3_color_normalization()
            
            # Step 4: Metadata Removal
            self.step_4_metadata_removal()
            
            # Step 5: Resolution Check
            self.step_5_resolution_check()
            
            # Step 6: Blurriness Detection
            self.step_6_blur_detection()
            
            # Step 7: Cropped/Partial Image Detection
            self.step_7_partial_image_detection()
            
            # Step 8: Brightness & Contrast Evaluation
            self.step_8_brightness_contrast_eval()
            
            # Step 9: Noise Detection
            self.step_9_noise_detection()
            
            # Step 10: Skew & Orientation Correction
            self.step_10_skew_correction()
            
            # Step 11: Auto-Resize & Padding
            self.step_11_resize_padding()
            
            # Step 12: Border/Whitespace Removal
            self.step_12_border_removal()
            
            # Step 13: Shadow & Background Cleaning
            self.step_13_shadow_cleaning()
            
            # Step 14: Sharpness Enhancement
            self.step_14_sharpness_enhancement()
            
            # Step 15: Contrast Enhancement
            self.step_15_contrast_enhancement()
            
            # Step 16: Color Normalization
            self.step_16_color_normalization()
            
            # Step 17: Thresholding/Binarization
            self.step_17_thresholding()
            
            # Step 18: Edge & Contour Processing
            self.step_18_edge_processing()
            
            # Step 19: Final Pixel Normalization
            self.step_19_pixel_normalization()
            
            # Step 20: Image Format Standardization
            self.step_20_format_standardization()
            
            # Step 21: Quality Score Logging
            self.step_21_quality_scoring()
            
            # Step 22: Audit Trail
            self.step_22_audit_trail()
            
            # Generate report
            self.generate_report()
            
            return self.processed_image, self.report
            
        except Exception as e:
            self.report['passed'] = False
            self.report['error'] = str(e)
            self.generate_report()
            raise e
    
    # Step Implementations
    def step_1_file_validation(self):
        """Ensure valid image format and convert if needed"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        ext = os.path.splitext(self.image_path)[1].lower()
        
        if ext not in valid_extensions:
            self.report['steps']['1_file_validation'] = {
                'status': 'failed',
                'message': f'Unsupported file format: {ext}'
            }
            self.report['passed'] = False
            raise ValueError(f'Unsupported file format: {ext}')
        
        self.report['steps']['1_file_validation'] = {
            'status': 'passed',
            'message': f'Supported format: {ext}'
        }
    
    def step_2_corruption_check(self):
        """Try opening the image to check for corruption"""
        try:
            self.image = Image.open(self.image_path)
            self.image.verify()  # Verify without loading
            self.image = Image.open(self.image_path)  # Reopen for processing
            
            # Also check with OpenCV
            cv2.imread(self.image_path)
            if cv2.imread(self.image_path) is None:
                raise ValueError("OpenCV couldn't read the image")
                
            self.report['steps']['2_corruption_check'] = {
                'status': 'passed',
                'message': 'Image is not corrupted'
            }
        except Exception as e:
            self.report['steps']['2_corruption_check'] = {
                'status': 'failed',
                'message': str(e)
            }
            self.report['passed'] = False
            raise ValueError(f'Corrupted image: {str(e)}')
    
    def step_3_color_normalization(self):
        """Convert to RGB or grayscale if needed"""
        # For this example, we'll convert to RGB unless it's a grayscale image
        if self.image.mode != 'RGB':
            if self.image.mode == 'L':
                self.grayscale = True
                self.report['steps']['3_color_normalization'] = {
                    'status': 'passed',
                    'message': 'Grayscale image maintained',
                    'original_mode': self.image.mode,
                    'new_mode': 'L'
                }
            else:
                self.image = self.image.convert('RGB')
                self.report['steps']['3_color_normalization'] = {
                    'status': 'passed',
                    'message': 'Converted to RGB',
                    'original_mode': self.image.mode,
                    'new_mode': 'RGB'
                }
        else:
            self.report['steps']['3_color_normalization'] = {
                'status': 'passed',
                'message': 'Already in RGB mode',
                'original_mode': 'RGB',
                'new_mode': 'RGB'
            }
    
    def step_4_metadata_removal(self):
        """Strip EXIF metadata"""
        try:
            # Remove using piexif
            if 'exif' in self.image.info:
                piexif.remove(self.image_path)
                self.image = Image.open(self.image_path)  # Reload without EXIF
                
            self.report['steps']['4_metadata_removal'] = {
                'status': 'passed',
                'message': 'EXIF metadata removed'
            }
        except Exception as e:
            self.report['steps']['4_metadata_removal'] = {
                'status': 'failed',
                'message': str(e)
            }
            self.report['flags'].append('Metadata removal failed')
    
    def step_5_resolution_check(self):
        """Ensure image meets minimum resolution"""
        min_resolution = (512, 512)
        width, height = self.image.size
        
        if width < min_resolution[0] or height < min_resolution[1]:
            self.report['steps']['5_resolution_check'] = {
                'status': 'failed',
                'message': f'Resolution {width}x{height} below minimum {min_resolution[0]}x{min_resolution[1]}',
                'width': width,
                'height': height,
                'min_width': min_resolution[0],
                'min_height': min_resolution[1]
            }
            self.report['passed'] = False
            raise ValueError('Image resolution too low')
        else:
            self.report['steps']['5_resolution_check'] = {
                'status': 'passed',
                'message': f'Resolution {width}x{height} meets minimum requirements',
                'width': width,
                'height': height
            }
    
    def step_6_blur_detection(self):
        """Detect blur using variance of Laplacian"""
        image_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        if self.grayscale:
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_threshold = 100  # Adjust based on your needs
        
        is_blurry = laplacian_var < blur_threshold
        
        self.report['steps']['6_blur_detection'] = {
            'status': 'passed',
            'is_blurry': is_blurry,
            'laplacian_variance': float(laplacian_var),
            'threshold': blur_threshold,
            'message': 'Image is blurry' if is_blurry else 'Image is sharp'
        }
        
        if is_blurry:
            self.report['flags'].append('Image is blurry')
    
    def step_7_partial_image_detection(self):
        """Detect abrupt borders or incomplete objects"""
        image_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Check for strong edges at borders
        border_thickness = 10
        top_edge = edges[:border_thickness, :].mean()
        bottom_edge = edges[-border_thickness:, :].mean()
        left_edge = edges[:, :border_thickness].mean()
        right_edge = edges[:, -border_thickness:].mean()
        
        edge_threshold = 10  # Adjust based on needs
        has_border_edges = (top_edge > edge_threshold or 
                          bottom_edge > edge_threshold or 
                          left_edge > edge_threshold or 
                          right_edge > edge_threshold)
        
        self.report['steps']['7_partial_image_detection'] = {
            'status': 'passed',
            'has_border_edges': has_border_edges,
            'top_edge_strength': float(top_edge),
            'bottom_edge_strength': float(bottom_edge),
            'left_edge_strength': float(left_edge),
            'right_edge_strength': float(right_edge),
            'threshold': edge_threshold,
            'message': 'Possible partial/cropped image' if has_border_edges else 'No border edges detected'
        }
        
        if has_border_edges:
            self.report['flags'].append('Possible partial/cropped image')
    
    def step_8_brightness_contrast_eval(self):
        """Evaluate brightness and contrast"""
        image_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        if self.grayscale:
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Brightness evaluation
        brightness = np.mean(gray)
        brightness_low_thresh = 30
        brightness_high_thresh = 220
        
        # Contrast evaluation
        contrast = gray.std()
        contrast_thresh = 40
        
        is_underexposed = brightness < brightness_low_thresh
        is_overexposed = brightness > brightness_high_thresh
        is_low_contrast = contrast < contrast_thresh
        
        self.report['steps']['8_brightness_contrast_evaluation'] = {
            'status': 'passed',
            'brightness': float(brightness),
            'contrast': float(contrast),
            'is_underexposed': is_underexposed,
            'is_overexposed': is_overexposed,
            'is_low_contrast': is_low_contrast,
            'brightness_low_threshold': brightness_low_thresh,
            'brightness_high_threshold': brightness_high_thresh,
            'contrast_threshold': contrast_thresh,
            'message': 'Brightness/contrast issues detected' if (is_underexposed or is_overexposed or is_low_contrast) else 'Brightness/contrast OK'
        }
        
        if is_underexposed:
            self.report['flags'].append('Image is underexposed')
        if is_overexposed:
            self.report['flags'].append('Image is overexposed')
        if is_low_contrast:
            self.report['flags'].append('Image has low contrast')
    
    def step_9_noise_detection(self):
        """Detect noise using image entropy"""
        image_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        if self.grayscale:
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Calculate entropy
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        
        # Calculate variance
        variance = gray.var()
        
        entropy_thresh = 5  # Adjust based on needs
        variance_thresh = 100  # Adjust based on needs
        
        is_noisy = entropy > entropy_thresh or variance < variance_thresh
        
        self.report['steps']['9_noise_detection'] = {
            'status': 'passed',
            'entropy': float(entropy),
            'variance': float(variance),
            'entropy_threshold': entropy_thresh,
            'variance_threshold': variance_thresh,
            'is_noisy': is_noisy,
            'message': 'Image is noisy' if is_noisy else 'Noise level acceptable'
        }
        
        if is_noisy:
            self.report['flags'].append('Image is noisy')
    
    def step_10_skew_correction(self):
        """Detect and correct skew using Hough Transform"""
        image_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 45:  # Only consider near-horizontal lines
                    angles.append(angle)
        
        if angles:
            median_angle = np.median(angles)
            if abs(median_angle) > 1:  # Only correct if angle > 1 degree
                self.image = Image.fromarray(imutils.rotate(np.array(self.image), angle=median_angle))
                
                self.report['steps']['10_skew_correction'] = {
                    'status': 'passed',
                    'corrected_angle': float(median_angle),
                    'message': f'Corrected skew by {median_angle:.2f} degrees'
                }
                return
        
        self.report['steps']['10_skew_correction'] = {
            'status': 'passed',
            'corrected_angle': 0.0,
            'message': 'No significant skew detected'
        }
    
    def step_11_resize_padding(self):
        """Resize to target size with padding to maintain aspect ratio"""
        target_size = (512, 512)
        img = np.array(self.image)
        
        # Get current and target dimensions
        h, w = img.shape[:2]
        target_h, target_w = target_size
        
        # Calculate ratio and new dimensions
        ratio = min(target_h / h, target_w / w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        
        # Resize image
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Calculate padding
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        pad_top = pad_h
        pad_bottom = target_h - new_h - pad_top
        pad_left = pad_w
        pad_right = target_w - new_w - pad_left
        
        # Add padding
        if len(img.shape) == 3:  # Color image
            color = [0, 0, 0]  # Black padding
        else:  # Grayscale
            color = 0
        
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, 
                                   cv2.BORDER_CONSTANT, value=color)
        
        self.image = Image.fromarray(padded)
        
        self.report['steps']['11_resize_padding'] = {
            'status': 'passed',
            'original_size': (w, h),
            'new_size': target_size,
            'resize_ratio': float(ratio),
            'padding': {
                'top': pad_top,
                'bottom': pad_bottom,
                'left': pad_left,
                'right': pad_right
            },
            'message': f'Resized to {target_size} with padding'
        }
    
    def step_12_border_removal(self):
        """Crop unnecessary whitespace or background edges"""
        image_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        if self.grayscale:
            gray = image_cv
        else:
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Threshold and find contours
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Crop the image
            cropped = image_cv[y:y+h, x:x+w]
            self.image = Image.fromarray(cropped)
            
            self.report['steps']['12_border_removal'] = {
                'status': 'passed',
                'crop_coordinates': {
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h
                },
                'message': f'Cropped to remove borders: {w}x{h}'
            }
        else:
            self.report['steps']['12_border_removal'] = {
                'status': 'passed',
                'message': 'No borders detected to remove'
            }
    
    def step_13_shadow_cleaning(self):
        """Remove shadows using morphology and thresholding"""
        try:
            image_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
            if self.grayscale:
                gray = image_cv
            else:
                gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            
            # Ensure we're working with 8-bit image
            if gray.dtype != np.uint8:
                gray = (gray * 255).astype('uint8')
            
            # Apply morphological operations to remove shadows
            dilated_img = cv2.dilate(gray, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(gray, bg_img)
            norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, 
                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            
            self.image = Image.fromarray(norm_img)
            self.grayscale = True  # Result is grayscale
            
            self.report['steps']['13_shadow_cleaning'] = {
                'status': 'passed',
                'message': 'Applied shadow removal using morphological operations'
            }
        except Exception as e:
            self.report['steps']['13_shadow_cleaning'] = {
                'status': 'failed',
                'message': str(e)
            }
            self.report['flags'].append('Shadow removal failed')
    
    def step_14_sharpness_enhancement(self):
        """Apply sharpening if image is blurry"""
        # Check if image was flagged as blurry
        if '6_blur_detection' in self.report['steps'] and \
           self.report['steps']['6_blur_detection']['is_blurry']:
            
            # Unsharp masking
            image_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
            if self.grayscale:
                gray = image_cv
            else:
                gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            
            gaussian = cv2.GaussianBlur(gray, (0, 0), 3)
            sharpened = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
            
            self.image = Image.fromarray(sharpened)
            
            self.report['steps']['14_sharpness_enhancement'] = {
                'status': 'passed',
                'message': 'Applied sharpening filter (unsharp mask)'
            }
        else:
            self.report['steps']['14_sharpness_enhancement'] = {
                'status': 'passed',
                'message': 'No sharpening applied (image not blurry)'
            }
    
    def step_15_contrast_enhancement(self):
        """Apply CLAHE for contrast enhancement"""
        try:
            image_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
            
            if self.grayscale:
                # Ensure image is 8-bit single channel
                if image_cv.dtype != np.uint8:
                    image_cv = (image_cv * 255).astype('uint8')
                if len(image_cv.shape) > 2:
                    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image_cv
                
                # Apply CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                self.image = Image.fromarray(enhanced)
            else:
                lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Ensure L channel is 8-bit
                if l.dtype != np.uint8:
                    l = (l * 255).astype('uint8')
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                
                # Merge channels and convert back to RGB
                merged = cv2.merge((cl, a, b))
                enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
                self.image = Image.fromarray(enhanced)
            
            self.report['steps']['15_contrast_enhancement'] = {
                'status': 'passed',
                'message': 'Applied CLAHE contrast enhancement'
            }
        except Exception as e:
            self.report['steps']['15_contrast_enhancement'] = {
                'status': 'failed',
                'message': f'CLAHE failed: {str(e)}'
            }
            self.report['flags'].append('Contrast enhancement failed')
            # Continue processing without contrast enhancement
    
    def step_16_color_normalization(self):
        """Normalize color channels"""
        if not self.grayscale:
            image_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
            
            # Convert to float and normalize
            normalized = image_cv.astype('float32') / 255.0
            
            # Normalize each channel
            for i in range(3):
                channel = normalized[:,:,i]
                channel_mean = channel.mean()
                channel_std = channel.std()
                normalized[:,:,i] = (channel - channel_mean) / channel_std
            
            # Scale back to 0-255
            normalized = ((normalized - normalized.min()) * (255.0 / (normalized.max() - normalized.min()))).astype('uint8')
            
            self.image = Image.fromarray(normalized)
            
            self.report['steps']['16_color_normalization'] = {
                'status': 'passed',
                'message': 'Applied per-channel mean/std normalization'
            }
        else:
            self.report['steps']['16_color_normalization'] = {
                'status': 'passed',
                'message': 'Skipped (grayscale image)'
            }
    
    def step_17_thresholding(self):
        """Apply adaptive thresholding for OCR preparation"""
        if self.grayscale:
            image_cv = np.array(self.image)
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(image_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            self.image = Image.fromarray(thresh)
            
            self.report['steps']['17_thresholding'] = {
                'status': 'passed',
                'message': 'Applied adaptive thresholding'
            }
        else:
            self.report['steps']['17_thresholding'] = {
                'status': 'passed',
                'message': 'Skipped (color image)'
            }
    
    def step_18_edge_processing(self):
        """Edge extraction for object detection or alignment"""
        image_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        if self.grayscale:
            gray = image_cv
        else:
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Store edge image for potential later use
        self.edge_image = edges
        
        self.report['steps']['18_edge_processing'] = {
            'status': 'passed',
            'message': 'Applied Canny edge detection'
        }
    
    def step_19_pixel_normalization(self):
        """Final pixel value normalization"""
        try:
            image_array = np.array(self.image)
            
            if len(image_array.shape) == 2:  # Grayscale
                normalized = image_array.astype('float32') / 255.0
            else:  # Color
                normalized = image_array.astype('float32') / 255.0
            
            self.processed_image = normalized
            
            self.report['steps']['19_pixel_normalization'] = {
                'status': 'passed',
                'message': 'Normalized pixel values to [0,1] range'
            }
        except Exception as e:
            self.report['steps']['19_pixel_normalization'] = {
                'status': 'failed',
                'message': str(e)
            }
            self.report['flags'].append('Pixel normalization failed')
            # Continue with unnormalized image
            self.processed_image = np.array(self.image)
    
    def step_20_format_standardization(self):
        """Save in standardized format"""
        output_path = os.path.join(self.output_dir, 
                                 f"processed_{os.path.basename(self.image_path)}")
        output_path = os.path.splitext(output_path)[0] + '.png'
        
        if self.processed_image is not None:
            # Convert back to 0-255 range for saving
            save_image = (self.processed_image * 255).astype('uint8')
            cv2.imwrite(output_path, cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR))
        else:
            self.image.save(output_path)
        
        self.report['output_path'] = output_path
        
        self.report['steps']['20_format_standardization'] = {
            'status': 'passed',
            'output_path': output_path,
            'message': f'Saved processed image as PNG to {output_path}'
        }
    
    def step_21_quality_scoring(self):
        """Calculate overall quality score (0-100)"""
        score = 100
        deductions = []
        
        # Deduct for blurriness
        if '6_blur_detection' in self.report['steps'] and self.report['steps']['6_blur_detection']['is_blurry']:
            blur_deduction = min(30, (100 - self.report['steps']['6_blur_detection']['laplacian_variance']) / 2)
            score -= blur_deduction
            deductions.append(f'Blurriness: -{blur_deduction:.1f}')
        
        # Deduct for brightness issues
        if '8_brightness_contrast_evaluation' in self.report['steps']:
            brightness_info = self.report['steps']['8_brightness_contrast_evaluation']
            if brightness_info['is_underexposed']:
                underexposed_deduction = min(20, (30 - brightness_info['brightness']) / 3)
                score -= underexposed_deduction
                deductions.append(f'Underexposed: -{underexposed_deduction:.1f}')
            if brightness_info['is_overexposed']:
                overexposed_deduction = min(20, (brightness_info['brightness'] - 220) / 3)
                score -= overexposed_deduction
                deductions.append(f'Overexposed: -{overexposed_deduction:.1f}')
            if brightness_info['is_low_contrast']:
                contrast_deduction = min(15, (40 - brightness_info['contrast']) / 2)
                score -= contrast_deduction
                deductions.append(f'Low contrast: -{contrast_deduction:.1f}')
        
        # Deduct for noise
        if '9_noise_detection' in self.report['steps'] and self.report['steps']['9_noise_detection']['is_noisy']:
            noise_deduction = min(15, (self.report['steps']['9_noise_detection']['entropy'] - 5) * 2)
            score -= noise_deduction
            deductions.append(f'Noise: -{noise_deduction:.1f}')
        
        # Deduct for possible cropping
        if '7_partial_image_detection' in self.report['steps'] and \
           self.report['steps']['7_partial_image_detection']['has_border_edges']:
            crop_deduction = 10
            score -= crop_deduction
            deductions.append(f'Possible cropping: -{crop_deduction:.1f}')
        
        # Ensure score is within bounds
        score = max(0, min(100, score))
        
        self.report['quality_score'] = score
        self.report['quality_deductions'] = deductions
        
        self.report['steps']['21_quality_scoring'] = {
            'status': 'passed',
            'score': score,
            'deductions': deductions,
            'message': f'Calculated quality score: {score}/100'
        }
    
    def step_22_audit_trail(self):
        """Final audit trail and flagging"""
        # Check if any flags were raised
        needs_review = len(self.report['flags']) > 0 or self.report['quality_score'] < 70
        
        self.report['needs_manual_review'] = needs_review
        
        self.report['steps']['22_audit_trail'] = {
            'status': 'passed',
            'needs_review': needs_review,
            'message': 'Audit trail completed' + (' - Needs manual review' if needs_review else '')
        }
    
    def generate_report(self):
        """Generate a detailed text report"""
        report_path = os.path.join(self.output_dir, 
                                 f"report_{os.path.splitext(os.path.basename(self.image_path))[0]}.txt")
        
        with open(report_path, 'w') as f:
            f.write(f"Image Quality Assessment Report\n")
            f.write(f"="*50 + "\n\n")
            
            f.write(f"File: {self.report['filename']}\n")
            f.write(f"Timestamp: {self.report['timestamp']}\n")
            f.write(f"Overall Status: {'PASSED' if self.report['passed'] else 'FAILED'}\n")
            f.write(f"Quality Score: {self.report.get('quality_score', 'N/A')}/100\n")
            f.write(f"Needs Manual Review: {'YES' if self.report.get('needs_manual_review', False) else 'NO'}\n\n")
            
            if 'quality_deductions' in self.report and self.report['quality_deductions']:
                f.write("Quality Deductions:\n")
                for deduction in self.report['quality_deductions']:
                    f.write(f"  - {deduction}\n")
                f.write("\n")
            
            if self.report['flags']:
                f.write("Flags/Warnings:\n")
                for flag in self.report['flags']:
                    f.write(f"  - {flag}\n")
                f.write("\n")
            
            f.write("Processing Steps:\n")
            f.write("-"*50 + "\n")
            for step_name, step_info in self.report['steps'].items():
                step_num = step_name.split('_')[0]
                step_title = ' '.join(step_name.split('_')[1:]).title()
                f.write(f"{step_num}. {step_title}:\n")
                f.write(f"    Status: {step_info['status']}\n")
                f.write(f"    Message: {step_info['message']}\n")
                
                # Add additional details if available
                detail_keys = [k for k in step_info.keys() if k not in ['status', 'message']]
                for key in detail_keys:
                    if isinstance(step_info[key], dict):
                        f.write(f"    {key.title()}:\n")
                        for subkey, value in step_info[key].items():
                            f.write(f"      {subkey}: {value}\n")
                    else:
                        f.write(f"    {key.title()}: {step_info[key]}\n")
                
                f.write("\n")
            
            if 'output_path' in self.report:
                f.write(f"Processed image saved to: {self.report['output_path']}\n")
        
        self.report['report_path'] = report_path
        print(f"Report generated at: {report_path}")

# Example usage
if __name__ == "__main__":
    # Initialize pipeline with input image path
    pipeline = ImageQualityPipeline(r"C:\Users\PC\Downloads\0012 1.jpg")
    
    # Run the pipeline
    try:
        processed_image, report = pipeline.run_pipeline()
        print(f"Processing completed. Quality score: {report['quality_score']}/100")
    except Exception as e:
        print(f"Processing failed: {str(e)}")