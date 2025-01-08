import cv2
import numpy as np
from math import cos, sin, acos



def create_eye_position_overlay(eye_left_right, eye_up_down, width=200, height=100):
    """
    Creates a visualization overlay showing eye position with normalized coordinates.
    
    Parameters:
    eye_left_right (float): Normalized position from -1 (left) to +1 (right)
    eye_up_down (float): Normalized position from -1 (up) to +1 (down)
    width (int): Width of the overlay image
    height (int): Height of the overlay image
    
    Returns:
    numpy.ndarray: BGR image showing eye position
    """
    # Create black background
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Calculate center and radius
    center = (width // 2, height // 2)
    radius = min(width, height) // 4  # Slightly smaller radius for better visibility
    
    # Draw reference circle
    cv2.circle(overlay, center, radius, (255, 255, 255), 1)
    
    # Clamp coordinates to [-1, 1] range
    eye_left_right = np.clip(eye_left_right, -1.0, 1.0)
    eye_up_down = np.clip(eye_up_down, -1.0, 1.0)
    
    # Calculate eye position
    x = int(center[0] + eye_left_right * radius)
    y = int(center[1] + eye_up_down * radius)
    
    # Draw crosshair at eye position
    crosshair_size = 5
    cv2.line(overlay, (x - crosshair_size, y), (x + crosshair_size, y), (0, 255, 255), 1)
    cv2.line(overlay, (x, y - crosshair_size), (x, y + crosshair_size), (0, 255, 255), 1)
    
    # Add coordinate text with improved formatting
    text = f"L/R: {eye_left_right:+.2f} U/D: {eye_up_down:+.2f}"
    cv2.putText(overlay, text, 
                (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,  # Smaller font size
                (255, 255, 255),
                1,
                cv2.LINE_AA)  # Anti-aliased text
    
    # Draw reference lines
    cv2.line(overlay, (center[0], 0), (center[0], height), (128, 128, 128), 1)
    cv2.line(overlay, (0, center[1]), (width, center[1]), (128, 128, 128), 1)
    
    return overlay

def create_diagnostic_overlay(frame, mask_image, eye_overlay, burn_image=None, metrics=None):
    import cv2
    import numpy as np
    
    height, width = frame.shape[:2]
    padding = 10
    diag_width = 144
    diag_height = 144
    
    # Create overlay with same number of channels as input frame
    overlay = frame.copy()
    
    def preprocess_image(image, target_width, target_height):
        """
        Preprocesses the input image to ensure it's a valid numpy array 
        and compatible with the frame dimensions.
        """
        if image is None:
            return None
        image = np.array(image)  # Ensure it's a numpy array
        if len(image.shape) < 2 or len(image.shape) > 3:
            raise ValueError("Input image must have 2 or 3 dimensions.")
        
        # Resize and ensure compatible datatype
        resized_image = cv2.resize(image, (target_width, target_height))
        if len(frame.shape) == 3 and len(resized_image.shape) == 2:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
        return resized_image.astype(frame.dtype)
    
    # Handle mask image (top left)
    if mask_image is not None:
        mask_resized = preprocess_image(mask_image, diag_width, diag_height)
        roi = overlay[padding:padding+diag_height, padding:padding+diag_width]
        if roi.shape[:2] == mask_resized.shape[:2]:
            overlay[padding:padding+diag_height, padding:padding+diag_width] = mask_resized
            cv2.putText(overlay, "Attention Mask", (padding, padding-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Handle burn image (top right)
    if burn_image is not None:
        burn_resized = preprocess_image(burn_image, diag_width, diag_height)
        roi = overlay[padding:padding+diag_height, width-padding-diag_width:width-padding]
        if roi.shape[:2] == burn_resized.shape[:2]:
            overlay[padding:padding+diag_height, width-padding-diag_width:width-padding] = burn_resized
            cv2.putText(overlay, "Visual Difference", 
                        (width-padding-diag_width, padding-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Handle eye overlay (bottom left)
    if eye_overlay is not None:
        eye_overlay = preprocess_image(eye_overlay, eye_overlay.shape[1], eye_overlay.shape[0])
        eye_h, eye_w = eye_overlay.shape[:2]
        if height-eye_h-padding > 0 and padding+eye_w <= width:
            roi = overlay[height-eye_h-padding:height-padding, padding:padding+eye_w]
            if roi.shape == eye_overlay.shape:
                overlay[height-eye_h-padding:height-padding, padding:padding+eye_w] = eye_overlay
    
    # Add metrics text
    if metrics:
        metrics_y = padding + diag_height + 20
        cv2.putText(overlay, f"Confidence: {metrics['confidence']:.2f}",
                    (padding, metrics_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        quat = metrics['head_quaternion']
        cv2.putText(overlay, f"Quaternion: [{quat[0]:.2f}, {quat[1]:.2f}, {quat[2]:.2f}, {quat[3]:.2f}]",
                    (padding, metrics_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Blend overlay with original frame
    alpha = 0.8
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

def draw_head_pose(frame, euler_angles, origin, scale=50):
    roll, pitch, yaw = euler_angles
    
    Rx = np.array([[1, 0, 0],
                   [0, cos(roll), -sin(roll)],
                   [0, sin(roll), cos(roll)]])
    
    Ry = np.array([[cos(pitch), 0, sin(pitch)],
                   [0, 1, 0],
                   [-sin(pitch), 0, cos(pitch)]])
    
    Rz = np.array([[cos(yaw), -sin(yaw), 0],
                   [sin(yaw), cos(yaw), 0],
                   [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    
    axes = np.array([[scale, 0, 0],
                     [0, scale, 0],
                     [0, 0, scale]])
    
    projected_axes = (R @ axes.T).T
    
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for i in range(3):
        cv2.line(frame, 
                 (int(origin[0]), int(origin[1])),
                 (int(origin[0] + projected_axes[i][0]), 
                  int(origin[1] + projected_axes[i][1])),
                 colors[i], 2)
def compute_visual_difference_heatmap(image1, image2):
    """
    Creates a visually distinct heatmap of differences between two images.

    Args:
        image1: First image (numpy array).
        image2: Second image (numpy array).

    Returns:
        diff_heatmap: A visually distinct heatmap highlighting differences.
    """
    import cv2
    import numpy as np

    # Ensure both images are the same size
    image1_resized = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

    # Compute the absolute difference per channel
    diff = cv2.absdiff(image1_resized, image2)

    # Amplify the differences by scaling
    diff_amplified = np.clip(diff * 4, 0, 255).astype(np.uint8)  # Scale differences up for visibility

    # Apply a heatmap to the difference
    diff_heatmap = cv2.applyColorMap(cv2.cvtColor(diff_amplified, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)

    # Optionally overlay the heatmap on the second image
    overlay = cv2.addWeighted(image2, 0.5, diff_heatmap, 0.5, 0)

    return overlay
