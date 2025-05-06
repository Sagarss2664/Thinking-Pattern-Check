import os
import cv2
import pyheif
from PIL import Image
import numpy as np
import pandas as pd
from google.cloud import vision
from datetime import datetime

class LetterExtractor:
    def __init__(self, credentials_path):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        self.client = vision.ImageAnnotatorClient()
        
    def heic_to_pil(self, heic_path):
        """Convert HEIC image to PIL Image"""
        heif_file = pyheif.read(heic_path)
        return Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data, 
            "raw", 
            heif_file.mode, 
            heif_file.stride
        )

    def process_image(self, image_path):
        """Process an image file and return OpenCV format"""
        if image_path.endswith('.heic'):
            img = self.heic_to_pil(image_path)
        else:
            img = Image.open(image_path)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def extract_letters(self, img_cv, output_dir):
        """Extract m, n, r letters from single image"""
        results = {
            'counts': {'m': 0, 'n': 0, 'r': 0},
            'extracted_files': [],
            'error': None
        }
        
        try:
            # Convert image to bytes for Google Vision API
            _, encoded_image = cv2.imencode('.jpg', img_cv)
            image = vision.Image(content=encoded_image.tobytes())
            
            # Detect text
            response = self.client.document_text_detection(
                image=image,
                image_context=vision.ImageContext(language_hints=["en-t-i0-handwrit"])
            )
            
            if response.error.message:
                raise Exception(response.error.message)
                
            # Process detected text
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            for symbol in word.symbols:
                                char = symbol.text.lower()
                                if char in ['m', 'n', 'r']:
                                    # Get bounding box
                                    vertices = symbol.bounding_box.vertices
                                    x_coords = [vertex.x for vertex in vertices]
                                    y_coords = [vertex.y for vertex in vertices]
                                    
                                    # Add padding
                                    padding = 2
                                    x1 = max(0, min(x_coords) - padding)
                                    y1 = max(0, min(y_coords) - padding)
                                    x2 = min(img_cv.shape[1], max(x_coords) + padding)
                                    y2 = min(img_cv.shape[0], max(y_coords) + padding)
                                    
                                    if x1 >= x2 or y1 >= y2:
                                        continue
                                    
                                    # Save extracted character
                                    cropped = img_cv[y1:y2, x1:x2]
                                    if cropped.size > 0:
                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        char_dir = os.path.join(output_dir, char)
                                        os.makedirs(char_dir, exist_ok=True)
                                        
                                        filename = f"{char}_{timestamp}_{results['counts'][char]}.png"
                                        save_path = os.path.join(char_dir, filename)
                                        cv2.imwrite(save_path, cropped)
                                        
                                        results['counts'][char] += 1
                                        results['extracted_files'].append(save_path)
        
        except Exception as e:
            results['error'] = str(e)
            
        return results

    def process_single_image(self, image_path, output_dir):
        """Full processing pipeline for one image"""
        os.makedirs(output_dir, exist_ok=True)
        img_cv = self.process_image(image_path)
        return self.extract_letters(img_cv, output_dir)