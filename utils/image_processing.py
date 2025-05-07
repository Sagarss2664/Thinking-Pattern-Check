import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import pytesseract
from pytesseract import Output

def initialize_tesseract():
    """Initialize Tesseract OCR with configuration"""
    try:
        # Configure Tesseract path (if needed)
        # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
        
        # Test Tesseract installation
        test_text = pytesseract.image_to_string(Image.new('RGB', (100, 100), color='white'))
        st.success("✅ Tesseract OCR initialized successfully")
    except Exception as e:
        st.error(f"""
            🔴 Tesseract Initialization Failed - Diagnostic Info:
            
            Common Solutions:
            1. Install Tesseract OCR: 
               - Linux: sudo apt install tesseract-ocr
               - Mac: brew install tesseract
               - Windows: Download installer
            2. Install language packs if needed
            3. Verify pytesseract can find the tesseract executable
            
            Full Error: {str(e)}
        """)
        raise

def heic_to_pil(heic_path):
    """Convert HEIC image to PIL Image"""
    import pyheif
    heif_file = pyheif.read(heic_path)
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride
    )
    return image

def process_image(image_path):
    """Process an image file and return OpenCV format"""
    if image_path.lower().endswith('.heic'):
        st.error("HEIC format not supported. Please upload JPG or PNG.")
        return None
    img = Image.open(image_path)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def extract_letters(img_cv, output_folder):
    """Extract 'm' letters from image using Tesseract OCR"""
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        # Initialize Tesseract
        initialize_tesseract()
        
        # Convert to grayscale and threshold for better OCR
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Get OCR data with character-level bounding boxes
        data = pytesseract.image_to_data(
            thresh, 
            output_type=Output.DICT,
            config='--psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        )
        
        m_count = 0
        m_images = []
        
        # Loop through each detected character
        for i in range(len(data['text'])):
            char = data['text'][i].lower()
            conf = int(data['conf'][i])
            
            # Only process confident 'm' detections
            if char == 'm' and conf > 60:
                (x, y, w, h) = (
                    data['left'][i],
                    data['top'][i],
                    data['width'][i],
                    data['height'][i]
                )
                
                # Add padding around the character
                padding = 2
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(img_cv.shape[1], x + w + padding)
                y2 = min(img_cv.shape[0], y + h + padding)
                
                if x1 >= x2 or y1 >= y2:
                    continue
                
                # Extract and save the 'm'
                cropped = img_cv[y1:y2, x1:x2]
                if cropped.size > 0:
                    m_path = os.path.join(output_folder, f'm_{m_count}.png')
                    cv2.imwrite(m_path, cropped)
                    m_images.append(m_path)
                    m_count += 1
        
        return m_images
    
    except Exception as e:
        st.error(f"Error in extract_letters: {str(e)}")
        raise