import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'heic'}
    GOOGLE_APPLICATION_CREDENTIALS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'google_Api.json')
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size

    @staticmethod
    def init_app(app):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'extracted_ms'), exist_ok=True)