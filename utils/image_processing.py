import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
import streamlit as st

# Optional for Windows users:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def process_image(image_path):
    """Process an image file and return OpenCV format"""
    if image_path.name.lower().endswith('.heic'):
        st.error("❌ HEIC format not supported. Please upload JPG or PNG.")
        return None
    img = Image.open(image_path).convert('RGB')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def extract_letters(img_cv, output_folder):
    """Extract 'm' letters from image using Tesseract OCR"""
    os.makedirs(output_folder, exist_ok=True)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    custom_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=mM'
    data = pytesseract.image_to_data(thresh, config=custom_config, output_type=pytesseract.Output.DICT)

    m_count = 0
    m_images = []

    for i in range(len(data['text'])):
        char = data['text'][i].strip().lower()
        if char == 'm':
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            padding = 2
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img_cv.shape[1], x + w + padding)
            y2 = min(img_cv.shape[0], y + h + padding)

            if x2 > x1 and y2 > y1:
                cropped = img_cv[y1:y2, x1:x2]
                if cropped.size > 0:
                    m_path = os.path.join(output_folder, f'm_{m_count}.png')
                    cv2.imwrite(m_path, cropped)
                    m_images.append(m_path)
                    m_count += 1

    return m_images
