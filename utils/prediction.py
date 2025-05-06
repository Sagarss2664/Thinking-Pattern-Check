import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
import math
from skimage.feature import hog
from skimage.morphology import skeletonize
from skimage.measure import find_contours
from scipy import ndimage
from tensorflow.keras.models import load_model

class MPredictor:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.load_components()

    def load_components(self):
        """Load all necessary components for prediction"""
        with open(os.path.join(self.model_dir, 'feature_scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)

        with open(os.path.join(self.model_dir, 'pca_model.pkl'), 'rb') as f:
            self.pca = pickle.load(f)

        with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'rb') as f:
            self.le = pickle.load(f)

        try:
            self.model = load_model(os.path.join(self.model_dir, 'm_thinking_model_final.keras'))
        except:
            self.model = load_model(os.path.join(self.model_dir, 'm_thinking_model_final.h5'))

        print("✅ All model components loaded successfully")

    def preprocess_image(self, image_path):
        """Preprocess the image for feature extraction"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not read the image file")

        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
        return img

    def extract_features(self, image):
        """Extract features from the preprocessed image"""
        h, w = image.shape
        if h != 128 or w != 128:
            scale = 128 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_resized = cv2.resize(image, (new_w, new_h))

            pad_top = (128 - new_h) // 2
            pad_bottom = 128 - new_h - pad_top
            pad_left = (128 - new_w) // 2
            pad_right = 128 - new_w - pad_left
            image = cv2.copyMakeBorder(img_resized, pad_top, pad_bottom, pad_left, pad_right,
                                     cv2.BORDER_CONSTANT, value=0)

        features = []

        # HOG features
        hog_feat = hog(image, orientations=12, pixels_per_cell=(16, 16),
                      cells_per_block=(2, 2), block_norm='L2-Hys')
        features.extend(hog_feat)

        # Aspect ratio and compactness
        h, w = image.shape
        features.append(w / h)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            features.append(cv2.contourArea(contours[0]) / (w * h))
        else:
            features.append(0)

        # Zernike moments
        zernike = self.zernike_moments(image, degree=8)
        features.extend(zernike)

        # Symmetry features
        features.extend(self.calculate_symmetry(image))

        # Slant angle
        features.append(self.calculate_slant_angle(image))

        # Stroke analysis
        features.extend(self.stroke_analysis(image))

        # Curvature analysis
        features.extend(self.curvature_analysis(image))

        # Pressure analysis
        features.extend(self.pressure_analysis(image))

        # Spatial distribution
        features.extend(self.spatial_distribution(image))

        # Hu moments
        features.extend(cv2.HuMoments(cv2.moments(image)).flatten())

        # Central moments
        m = cv2.moments(image)
        features.extend([m['mu20'], m['mu11'], m['mu02'], m['mu30'], m['mu21'], m['mu12'], m['mu03']])

        # Contour-based features
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            features.append(cv2.arcLength(cnt, True))
            features.append(cv2.contourArea(cnt))
            rect = cv2.minAreaRect(cnt)
            features.append(rect[1][0] / rect[1][1])
            features.append(cv2.matchShapes(cnt, cv2.convexHull(cnt), cv2.CONTOURS_MATCH_I1, 0))
            ellipse = cv2.fitEllipse(cnt)
            features.append(ellipse[1][0] / ellipse[1][1])
        else:
            features.extend([0]*5)

        # Fourier descriptors
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            fd = self.fourier_descriptors(cnt, num_descriptors=20)
            features.extend(fd)
        else:
            features.extend([0]*20)

        return np.array(features)

    # Feature extraction helper methods
    def zernike_moments(self, image, degree=8):
        image = (image > 0).astype(np.uint8)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros((degree+1)*(degree+2)//2)

        cnt = max(contours, key=cv2.contourArea)
        moments = cv2.moments(cnt)
        if moments['m00'] == 0:
            return np.zeros((degree+1)*(degree+2)//2)

        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        radius = math.sqrt(moments['mu20'] + moments['mu02'])
        if radius < 1e-6:
            return np.zeros((degree+1)*(degree+2)//2)

        zernike = []
        for n in range(degree+1):
            for m in range(n+1):
                if (n - m) % 2 == 0:
                    V = 0
                    for point in cnt:
                        x, y = point[0]
                        x = (x - cx) / radius
                        y = (y - cy) / radius
                        r = math.sqrt(x**2 + y**2)
                        if r > 1:
                            continue
                        theta = math.atan2(y, x)
                        R = 0
                        for s in range(0, (n-m)//2 + 1):
                            R += ((-1)**s * math.factorial(n-s) /
                                 (math.factorial(s) * math.factorial((n+m)//2 - s) *
                                  math.factorial((n-m)//2 - s))) * r**(n - 2*s)
                        V += R * np.exp(-1j * m * theta)
                    V *= (n + 1) / np.pi
                    zernike.append(np.abs(V))
        return np.array(zernike)

    def calculate_symmetry(self, image):
        flipped_v = np.fliplr(image)
        vertical_sym = np.sum(image == flipped_v) / image.size

        flipped_h = np.flipud(image)
        horizontal_sym = np.sum(image == flipped_h) / image.size

        flipped_d1 = np.rot90(np.fliplr(image))
        diagonal_sym1 = np.sum(image == flipped_d1) / image.size

        flipped_d2 = np.rot90(np.flipud(image))
        diagonal_sym2 = np.sum(image == flipped_d2) / image.size

        return vertical_sym, horizontal_sym, diagonal_sym1, diagonal_sym2

    def calculate_slant_angle(self, image):
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)

        if lines is None:
            return 0.0

        angles = []
        for line in lines[:min(10, len(lines))]:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
                if -75 <= angle <= 75:
                    angles.append(angle)

        if not angles:
            return 0.0

        return np.median(angles)

    def stroke_analysis(self, image):
        skeleton = skeletonize(image > 0)
        distance_map = ndimage.distance_transform_edt(image > 0)
        stroke_width = distance_map * skeleton

        endpoints = 0
        junctions = 0
        for i in range(1, skeleton.shape[0]-1):
            for j in range(1, skeleton.shape[1]-1):
                if skeleton[i,j]:
                    neighbors = skeleton[i-1:i+2, j-1:j+2].sum() - 1
                    if neighbors == 1:
                        endpoints += 1
                    elif neighbors >= 3:
                        junctions += 1

        width_stats = [np.mean(stroke_width[skeleton]), np.std(stroke_width[skeleton]),
                       np.max(stroke_width[skeleton])] if np.any(skeleton) else [0, 0, 0]

        return endpoints, junctions, *width_stats

    def curvature_analysis(self, image):
        contours = find_contours(image > 0, 0.5)
        if not contours:
            return 0.0, 0.0

        contour = max(contours, key=len)
        if len(contour) < 10:
            return 0.0, 0.0

        curvatures = []
        for i in range(1, len(contour)-1):
            p1 = contour[i-1]
            p2 = contour[i]
            p3 = contour[i+1]

            v1 = p1 - p2
            v2 = p3 - p2

            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            curvatures.append(angle)

        if not curvatures:
            return 0.0, 0.0

        return np.mean(curvatures), np.std(curvatures)

    def pressure_analysis(self, image):
        if np.max(image) == 0:
            return 0.0, 0.0, 0.0

        normalized = image / np.max(image)
        return np.mean(normalized), np.std(normalized), np.max(normalized) - np.min(normalized)

    def spatial_distribution(self, image):
        y_proj = np.sum(image > 0, axis=1)
        x_proj = np.sum(image > 0, axis=0)

        y_center = np.argmax(y_proj)
        x_center = np.argmax(x_proj)

        y_balance = np.sum(y_proj[:y_center]) / np.sum(y_proj[y_center:]) if np.sum(y_proj[y_center:]) > 0 else 1.0
        x_balance = np.sum(x_proj[:x_center]) / np.sum(x_proj[x_center:]) if np.sum(x_proj[x_center:]) > 0 else 1.0

        return y_balance, x_balance, y_center/image.shape[0], x_center/image.shape[1]

    def fourier_descriptors(self, contour, num_descriptors=20):
        complex_contour = np.empty(len(contour), dtype=complex)
        for i in range(len(contour)):
            complex_contour[i] = contour[i][0][0] + 1j * contour[i][0][1]

        fd = np.fft.fft(complex_contour)
        fd = np.abs(fd)

        if len(fd) > 1:
            fd = fd[1:min(num_descriptors+1, len(fd))] / fd[0]
        else:
            fd = np.zeros(num_descriptors)

        if len(fd) < num_descriptors:
            fd = np.pad(fd, (0, num_descriptors - len(fd)), 'constant')

        return fd

    def predict(self, image_path):
        """Make prediction on a new image"""
        try:
            img = self.preprocess_image(image_path)
            features = self.extract_features(img)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            features_pca = self.pca.transform(features_scaled)
            pred = self.model.predict(features_pca, verbose=0)
            pred_class = self.le.inverse_transform([np.argmax(pred)])[0]
            return pred_class, pred[0]
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None, None