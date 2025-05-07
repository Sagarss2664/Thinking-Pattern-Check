import os
import cv2
import pyheif
import numpy as np
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
import streamlit as st
import json

def get_google_vision_client():
    """Initialize Google Vision client with credentials"""
    try:
        # Option 1: Streamlit Secrets (properly formatted)
        if 'gcp_service_account' in st.secrets:
            creds_info = dict(st.secrets["gcp_service_account"])
            credentials = service_account.Credentials.from_service_account_info(creds_info)
            return vision.ImageAnnotatorClient(credentials=credentials)
        
        # Option 2: Local JSON file
        credential_path = "../credentials/google_Api.json"
        if os.path.exists(credential_path):
            credentials = service_account.Credentials.from_service_account_file(credential_path)
            return vision.ImageAnnotatorClient(credentials=credentials)
        
        # Option 3: Environment variable (for GCP deployments)
        if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
            return vision.ImageAnnotatorClient()
            
        raise Exception("No valid Google Cloud credentials found")
        
    except Exception as e:
        st.error(f"❌ Failed to initialize Google Vision client: {str(e)}")
        st.error("Please check your authentication setup")
        raise

def heic_to_pil(heic_path):
    """Convert HEIC image to PIL Image"""
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
    """Extract 'm' letters from image using Google Vision API"""
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        client = get_google_vision_client()
        _, encoded_image = cv2.imencode('.jpg', img_cv)
        content = encoded_image.tobytes()

        image = vision.Image(content=content)
        context = vision.ImageContext(language_hints=["en-t-i0-handwrit"])
        response = client.document_text_detection(image=image, image_context=context)

        m_count = 0
        m_images = []

        if response.error.message:
            raise Exception(f'Google Vision API error: {response.error.message}')

        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        for symbol in word.symbols:
                            char = symbol.text.lower()
                            if char == 'm':
                                vertices = symbol.bounding_box.vertices
                                x_coords = [vertex.x for vertex in vertices]
                                y_coords = [vertex.y for vertex in vertices]

                                x1, x2 = min(x_coords), max(x_coords)
                                y1, y2 = min(y_coords), max(y_coords)

                                padding = 2
                                x1 = max(0, x1 - padding)
                                y1 = max(0, y1 - padding)
                                x2 = min(img_cv.shape[1], x2 + padding)
                                y2 = min(img_cv.shape[0], y2 + padding)

                                if x1 >= x2 or y1 >= y2:
                                    continue

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