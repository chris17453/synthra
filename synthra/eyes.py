import cv2
import numpy as np
from .config import face_mesh

def mask_eyes(image):
    """Mask out the eye regions in the image"""
    # Convert to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    
    if not results.multi_face_landmarks:
        return image
    
    # Create a mask for the eyes
    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    h, w = image.shape[:2]
    
    # Eye landmark indices for MediaPipe Face Mesh
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    
    landmarks = results.multi_face_landmarks[0]
    
    # Function to draw filled polygon for eye region
    def draw_eye_region(landmarks, indices):
        points = []
        for idx in indices:
            point = landmarks.landmark[idx]
            points.append((int(point.x * w), int(point.y * h)))
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 0)
    
    # Draw both eye regions on the mask
    draw_eye_region(landmarks, LEFT_EYE)
    draw_eye_region(landmarks, RIGHT_EYE)
    
    # Add some padding around the eyes
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    
    # Apply the mask to the image
    masked_image = image.copy()
    masked_image[mask == 0] = 0  # Set eye regions to gray (128)
    
    return masked_image
