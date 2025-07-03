import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from PIL.ExifTags import TAGS
import json
import datetime
from pathlib import Path
from skimage import filters, measure, morphology, exposure
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
import logging
from typing import Tuple, Dict, List, Optional
import shutil

class ImageQualityPipeline:
    """
    Comprehensive image quality assessment and preprocessing pipeline
    """
    
    def __init__(self, output_dir: str = "pipeline_output"):
        self.output_dir = Path(output_dir)
        self.setup_directories()
        self.setup_logging()
        
        # Quality thresholds
        self.min_resolution = (512, 512)
        self.blur_threshold = 100.0
        self.brightness_range = (50, 200)
        self.contrast_threshold = 30
        self.noise_threshold = 0.8
        
    def setup_directories(self):
        """Create output directory structure"""
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "processed_images").mkdir(exist_ok=True)
        (self.output_dir / "rejected_images").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "audit_trail").mkdir(exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "pipeline.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def step_1_file_validation(self, image_path: str) -> Dict:
        """Step 1: File Validation"""
        result = {"step": 1, "name": "File Validation", "passed": False, "details": ""}
        
        try:
            # Check file extension
            valid_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
            file_ext = Path(image_path).suffix.lower()
            
            if file_ext not in valid_extensions:
                result["details"] = f"Invalid format: {file_ext}"
                return result
            
            # Try to open with PIL
            with Image.open(image_path) as img:
                result["passed"] = True
                result["details"] = f"Valid {img.format} image"
                result["format"] = img.format
                result["size"] = img.size
                
        except Exception as e:
            result["details"] = f"File validation error: {str(e)}"
        
        return result
    
    def step_2_corruption_check(self, image_path: str) -> Dict:
        """Step 2: Corruption Check"""
        result = {"step": 2, "name": "Corruption Check", "passed": False, "details": ""}
        
        try:
            # Try OpenCV
            cv_img = cv2.imread(image_path)
            if cv_img is None:
                result["details"] = "OpenCV cannot read image"
                return result
            
            # Try PIL
            with Image.open(image_path) as pil_img:
                pil_img.verify()
            
            result["passed"] = True
            result["details"] = "Image is not corrupted"
            
        except Exception as e:
            result["details"] = f"Corruption detected: {str(e)}"
        
        return result
    
    def step_3_color_mode_normalization(self, image_path: str) -> Tuple[Dict, Optional[np.ndarray]]:
        """Step 3: Color Mode Normalization"""
        result = {"step": 3, "name": "Color Mode Normalization", "passed": False, "details": ""}
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                result["details"] = "Cannot read image for color normalization"
                return result, None
            
            # Convert BGR to RGB
            if len(img.shape) == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result["details"] = "Converted to RGB"
            else:
                img_rgb = img
                result["details"] = "Already grayscale"
            
            result["passed"] = True
            result["channels"] = img_rgb.shape[2] if len(img_rgb.shape) == 3 else 1
            
            return result, img_rgb
            
        except Exception as e:
            result["details"] = f"Color normalization error: {str(e)}"
            return result, None
    
    def step_4_metadata_removal(self, image_path: str) -> Dict:
        """Step 4: Metadata Removal"""
        result = {"step": 4, "name": "Metadata Removal", "passed": False, "details": ""}
        
        try:
            with Image.open(image_path) as img:
                # Check for EXIF data
                exif_data = img._getexif()
                if exif_data:
                    result["details"] = f"EXIF data found: {len(exif_data)} entries"
                    # Create clean image without EXIF
                    clean_img = Image.new(img.mode, img.size)
                    clean_img.putdata(list(img.getdata()))
                    result["cleaned"] = True
                else:
                    result["details"] = "No EXIF data found"
                    result["cleaned"] = False
                
                result["passed"] = True
                
        except Exception as e:
            result["details"] = f"Metadata removal error: {str(e)}"
        
        return result
    
    def step_5_resolution_check(self, img: np.ndarray) -> Dict:
        """Step 5: Resolution Check"""
        result = {"step": 5, "name": "Resolution Check", "passed": False, "details": ""}
        
        try:
            height, width = img.shape[:2]
            result["resolution"] = (width, height)
            
            if width >= self.min_resolution[0] and height >= self.min_resolution[1]:
                result["passed"] = True
                result["details"] = f"Resolution {width}x{height} meets minimum requirement"
            else:
                result["details"] = f"Resolution {width}x{height} below minimum {self.min_resolution}"
                
        except Exception as e:
            result["details"] = f"Resolution check error: {str(e)}"
        
        return result
    
    def step_6_blurriness_detection(self, img: np.ndarray) -> Dict:
        """Step 6: Blurriness Detection"""
        result = {"step": 6, "name": "Blurriness Detection", "passed": False, "details": ""}
        
        try:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            
            # Variance of Laplacian
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            result["laplacian_variance"] = float(laplacian_var)
            
            if laplacian_var > self.blur_threshold:
                result["passed"] = True
                result["details"] = f"Image is sharp (variance: {laplacian_var:.2f})"
            else:
                result["details"] = f"Image is blurry (variance: {laplacian_var:.2f})"
                
        except Exception as e:
            result["details"] = f"Blurriness detection error: {str(e)}"
        
        return result
    
    def step_7_cropped_detection(self, img: np.ndarray) -> Dict:
        """Step 7: Cropped/Partial Image Detection"""
        result = {"step": 7, "name": "Cropped/Partial Image Detection", "passed": False, "details": ""}
        
        try:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Check edges near borders
            border_width = 10
            h, w = edges.shape
            
            top_edges = np.sum(edges[:border_width, :])
            bottom_edges = np.sum(edges[-border_width:, :])
            left_edges = np.sum(edges[:, :border_width])
            right_edges = np.sum(edges[:, -border_width:])
            
            border_edge_ratio = (top_edges + bottom_edges + left_edges + right_edges) / np.sum(edges)
            
            result["border_edge_ratio"] = float(border_edge_ratio)
            
            if border_edge_ratio < 0.3:  # Threshold for cropped detection
                result["passed"] = True
                result["details"] = f"Image appears complete (border edge ratio: {border_edge_ratio:.3f})"
            else:
                result["details"] = f"Image may be cropped (border edge ratio: {border_edge_ratio:.3f})"
                
        except Exception as e:
            result["details"] = f"Cropped detection error: {str(e)}"
        
        return result
    
    def step_8_brightness_contrast_evaluation(self, img: np.ndarray) -> Dict:
        """Step 8: Brightness & Contrast Evaluation"""
        result = {"step": 8, "name": "Brightness & Contrast Evaluation", "passed": False, "details": ""}
        
        try:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            
            # Calculate brightness (mean)
            brightness = np.mean(gray)
            
            # Calculate contrast (standard deviation)
            contrast = np.std(gray)
            
            result["brightness"] = float(brightness)
            result["contrast"] = float(contrast)
            
            brightness_ok = self.brightness_range[0] <= brightness <= self.brightness_range[1]
            contrast_ok = contrast >= self.contrast_threshold
            
            if brightness_ok and contrast_ok:
                result["passed"] = True
                result["details"] = f"Good brightness ({brightness:.1f}) and contrast ({contrast:.1f})"
            else:
                issues = []
                if not brightness_ok:
                    issues.append(f"brightness {brightness:.1f} outside range {self.brightness_range}")
                if not contrast_ok:
                    issues.append(f"low contrast {contrast:.1f}")
                result["details"] = f"Issues: {', '.join(issues)}"
                
        except Exception as e:
            result["details"] = f"Brightness/contrast evaluation error: {str(e)}"
        
        return result
    
    def step_9_noise_detection(self, img: np.ndarray) -> Dict:
        """Step 9: Noise Detection"""
        result = {"step": 9, "name": "Noise Detection", "passed": False, "details": ""}
        
        try:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            
            # Calculate image entropy as noise measure
            hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
            hist = hist / hist.sum()  # Normalize
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            result["entropy"] = float(entropy)
            
            if entropy < self.noise_threshold:
                result["passed"] = True
                result["details"] = f"Low noise (entropy: {entropy:.3f})"
            else:
                result["details"] = f"High noise detected (entropy: {entropy:.3f})"
                
        except Exception as e:
            result["details"] = f"Noise detection error: {str(e)}"
        
        return result
    
    def step_10_skew_correction(self, img: np.ndarray) -> Tuple[Dict, np.ndarray]:
        """Step 10: Skew & Orientation Correction"""
        result = {"step": 10, "name": "Skew & Orientation Correction", "passed": False, "details": ""}
        corrected_img = img.copy()
        
        try:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Hough line detection
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = np.degrees(theta) - 90
                    angles.append(angle)
                
                # Find dominant angle
                median_angle = np.median(angles)
                
                if abs(median_angle) > 1:  # Only correct if significant skew
                    # Rotate image
                    h, w = gray.shape
                    center = (w // 2, h // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    corrected_img = cv2.warpAffine(img, rotation_matrix, (w, h))
                    
                    result["details"] = f"Corrected skew by {median_angle:.2f} degrees"
                    result["angle_corrected"] = float(median_angle)
                else:
                    result["details"] = f"No significant skew detected ({median_angle:.2f} degrees)"
                    result["angle_corrected"] = 0.0
            else:
                result["details"] = "No lines detected for skew correction"
                result["angle_corrected"] = 0.0
            
            result["passed"] = True
            
        except Exception as e:
            result["details"] = f"Skew correction error: {str(e)}"
        
        return result, corrected_img
    
    def step_11_resize_padding(self, img: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> Tuple[Dict, np.ndarray]:
        """Step 11: Auto-Resize & Padding"""
        result = {"step": 11, "name": "Auto-Resize & Padding", "passed": False, "details": ""}
        
        try:
            h, w = img.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scaling factor to maintain aspect ratio
            scale = min(target_w / w, target_h / h)
            
            # Calculate new dimensions
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create padded image
            if len(img.shape) == 3:
                padded_img = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
            else:
                padded_img = np.zeros((target_h, target_w), dtype=img.dtype)
            
            # Calculate padding
            pad_x = (target_w - new_w) // 2
            pad_y = (target_h - new_h) // 2
            
            # Place resized image in center
            if len(img.shape) == 3:
                padded_img[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_img
            else:
                padded_img[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_img
            
            result["passed"] = True
            result["details"] = f"Resized from {w}x{h} to {new_w}x{new_h}, padded to {target_w}x{target_h}"
            result["scale_factor"] = float(scale)
            result["padding"] = (pad_x, pad_y)
            
            return result, padded_img
            
        except Exception as e:
            result["details"] = f"Resize/padding error: {str(e)}"
            return result, img
    
    def calculate_quality_score(self, results: List[Dict]) -> float:
        """Step 21: Calculate overall quality score"""
        total_score = 0
        max_score = 0
        
        # Weight different checks
        weights = {
            1: 10,  # File validation
            2: 10,  # Corruption check
            5: 8,   # Resolution
            6: 8,   # Blurriness
            7: 6,   # Cropped detection
            8: 8,   # Brightness/contrast
            9: 6,   # Noise detection
        }
        
        for result in results:
            step = result["step"]
            if step in weights:
                max_score += weights[step]
                if result["passed"]:
                    total_score += weights[step]
        
        return (total_score / max_score * 100) if max_score > 0 else 0
    
    def process_image(self, image_path: str) -> Dict:
        """Process a single image through the entire pipeline"""
        start_time = datetime.datetime.now()
        results = []
        
        self.logger.info(f"Processing image: {image_path}")
        
        # Step 1: File Validation
        result_1 = self.step_1_file_validation(image_path)
        results.append(result_1)
        if not result_1["passed"]:
            return self.create_final_report(image_path, results, start_time, "REJECTED")
        
        # Step 2: Corruption Check
        result_2 = self.step_2_corruption_check(image_path)
        results.append(result_2)
        if not result_2["passed"]:
            return self.create_final_report(image_path, results, start_time, "REJECTED")
        
        # Step 3: Color Mode Normalization
        result_3, img_rgb = self.step_3_color_mode_normalization(image_path)
        results.append(result_3)
        if img_rgb is None:
            return self.create_final_report(image_path, results, start_time, "REJECTED")
        
        # Step 4: Metadata Removal
        result_4 = self.step_4_metadata_removal(image_path)
        results.append(result_4)
        
        # Step 5: Resolution Check
        result_5 = self.step_5_resolution_check(img_rgb)
        results.append(result_5)
        
        # Step 6: Blurriness Detection
        result_6 = self.step_6_blurriness_detection(img_rgb)
        results.append(result_6)
        
        # Step 7: Cropped/Partial Image Detection
        result_7 = self.step_7_cropped_detection(img_rgb)
        results.append(result_7)
        
        # Step 8: Brightness & Contrast Evaluation
        result_8 = self.step_8_brightness_contrast_evaluation(img_rgb)
        results.append(result_8)
        
        # Step 9: Noise Detection
        result_9 = self.step_9_noise_detection(img_rgb)
        results.append(result_9)
        
        # Step 10: Skew & Orientation Correction
        result_10, corrected_img = self.step_10_skew_correction(img_rgb)
        results.append(result_10)
        
        # Step 11: Auto-Resize & Padding
        result_11, final_img = self.step_11_resize_padding(corrected_img)
        results.append(result_11)
        
        # Calculate quality score
        quality_score = self.calculate_quality_score(results)
        
        # Determine final status
        critical_failures = [r for r in results if r["step"] in [1, 2, 5] and not r["passed"]]
        status = "REJECTED" if critical_failures else "PROCESSED"
        
        # Save processed image
        output_path = self.save_processed_image(image_path, final_img, status)
        
        # Create final report
        final_report = self.create_final_report(image_path, results, start_time, status, quality_score, output_path)
        
        return final_report
    
    def save_processed_image(self, original_path: str, processed_img: np.ndarray, status: str) -> str:
        """Save the processed image to appropriate folder"""
        filename = Path(original_path).stem + "_processed.png"
        
        if status == "PROCESSED":
            output_path = self.output_dir / "processed_images" / filename
        else:
            output_path = self.output_dir / "rejected_images" / filename
        
        # Convert RGB to BGR for OpenCV saving
        if len(processed_img.shape) == 3:
            save_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
        else:
            save_img = processed_img
        
        cv2.imwrite(str(output_path), save_img)
        return str(output_path)
    
    def create_final_report(self, image_path: str, results: List[Dict], start_time: datetime.datetime, 
                           status: str, quality_score: float = 0, output_path: str = "") -> Dict:
        """Create comprehensive report for the processed image"""
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        report = {
            "image_path": image_path,
            "filename": Path(image_path).name,
            "processing_time_seconds": processing_time,
            "timestamp": start_time.isoformat(),
            "status": status,
            "quality_score": quality_score,
            "output_path": output_path,
            "step_results": results,
            "summary": {
                "total_steps": len(results),
                "passed_steps": sum(1 for r in results if r["passed"]),
                "failed_steps": sum(1 for r in results if not r["passed"]),
                "critical_failures": [r["name"] for r in results if r["step"] in [1, 2, 5] and not r["passed"]]
            }
        }
        
        return report
    
    def save_report(self, report: Dict, format: str = "both") -> str:
        """Save report as text and/or JSON"""
        filename = Path(report["filename"]).stem
        
        if format in ["text", "both"]:
            text_path = self.output_dir / "reports" / f"{filename}_report.txt"
            self.save_text_report(report, text_path)
        
        if format in ["json", "both"]:
            json_path = self.output_dir / "reports" / f"{filename}_report.json"
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return str(text_path) if format == "text" else str(json_path)
    
    def save_text_report(self, report: Dict, output_path: str):
        """Save human-readable text report"""
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("IMAGE QUALITY ASSESSMENT REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Image: {report['filename']}\n")
            f.write(f"Processing Time: {report['processing_time_seconds']:.2f} seconds\n")
            f.write(f"Timestamp: {report['timestamp']}\n")
            f.write(f"Status: {report['status']}\n")
            f.write(f"Quality Score: {report['quality_score']:.1f}/100\n")
            f.write(f"Output Path: {report['output_path']}\n\n")
            
            f.write("STEP-BY-STEP RESULTS:\n")
            f.write("-"*50 + "\n")
            
            for result in report['step_results']:
                status_icon = "✓" if result['passed'] else "✗"
                f.write(f"{status_icon} Step {result['step']}: {result['name']}\n")
                f.write(f"   Details: {result['details']}\n")
                
                # Add specific metrics if available
                metrics = {k: v for k, v in result.items() if k not in ['step', 'name', 'passed', 'details']}
                if metrics:
                    f.write(f"   Metrics: {metrics}\n")
                f.write("\n")
            
            f.write("SUMMARY:\n")
            f.write("-"*20 + "\n")
            f.write(f"Total Steps: {report['summary']['total_steps']}\n")
            f.write(f"Passed Steps: {report['summary']['passed_steps']}\n")
            f.write(f"Failed Steps: {report['summary']['failed_steps']}\n")
            
            if report['summary']['critical_failures']:
                f.write(f"Critical Failures: {', '.join(report['summary']['critical_failures'])}\n")
            
            f.write("\n" + "="*80 + "\n")
    
    def process_batch(self, input_folder: str, file_extensions: List[str] = None) -> List[Dict]:
        """Process all images in a folder"""
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
        
        input_path = Path(input_folder)
        image_files = []
        
        for ext in file_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        reports = []
        for image_file in image_files:
            try:
                report = self.process_image(str(image_file))
                reports.append(report)
                self.save_report(report)
                self.logger.info(f"Completed: {image_file.name} - Status: {report['status']}")
            except Exception as e:
                self.logger.error(f"Error processing {image_file.name}: {str(e)}")
        
        # Create batch summary
        self.create_batch_summary(reports)
        
        return reports
    
    def create_batch_summary(self, reports: List[Dict]):
        """Create a summary report for batch processing"""
        total_images = len(reports)
        processed_images = sum(1 for r in reports if r['status'] == 'PROCESSED')
        rejected_images = sum(1 for r in reports if r['status'] == 'REJECTED')
        
        avg_quality = np.mean([r['quality_score'] for r in reports])
        avg_processing_time = np.mean([r['processing_time_seconds'] for r in reports])
        
        summary = {
            "batch_summary": {
                "total_images": total_images,
                "processed_images": processed_images,
                "rejected_images": rejected_images,
                "success_rate": (processed_images / total_images * 100) if total_images > 0 else 0,
                "average_quality_score": avg_quality,
                "average_processing_time": avg_processing_time
            },
            "timestamp": datetime.datetime.now().isoformat(),
            "reports": reports
        }
        
        # Save batch summary
        summary_path = self.output_dir / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save text summary
        text_summary_path = self.output_dir / "batch_summary.txt"
        with open(text_summary_path, 'w') as f:
            f.write("BATCH PROCESSING SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total Images: {total_images}\n")
            f.write(f"Processed: {processed_images}\n")
            f.write(f"Rejected: {rejected_images}\n")
            f.write(f"Success Rate: {(processed_images / total_images * 100):.1f}%\n")
            f.write(f"Average Quality Score: {avg_quality:.1f}/100\n")
            f.write(f"Average Processing Time: {avg_processing_time:.2f} seconds\n\n")
            
            f.write("INDIVIDUAL RESULTS:\n")
            f.write("-"*30 + "\n")
            for report in reports:
                f.write(f"{report['filename']}: {report['status']} (Quality: {report['quality_score']:.1f})\n")
        
        self.logger.info(f"Batch summary saved to {summary_path}")


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = ImageQualityPipeline("output")
    
    # Process single image
    # report = pipeline.process_image("path/to/your/image.jpg")
    # pipeline.save_report(report)
    
    # Process batch of images
    # reports = pipeline.process_batch("path/to/input/folder")
    
    print("Image Quality Assessment Pipeline initialized.")
    print("Usage:")
    print("1. Process single image: pipeline.process_image('image_path')")
    print("2. Process batch: pipeline.process_batch('folder_path')")
    print("3. Reports will be saved in the output directory")