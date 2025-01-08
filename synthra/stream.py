import os
import cv2
import pyvirtualcam
from PIL import Image
import numpy as np
import logging

from .config import mp
from .eyes import mask_eyes
from .head import HeadTracker, detect_and_crop_head
from .overlay import create_diagnostic_overlay, create_eye_position_overlay, draw_head_pose, compute_visual_difference_heatmap
from .transform import transform_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_stream(model, 
                input_device,
                output_device,
                width,
                height):
    
    # Log input configuration
    logger.info(f"Initializing video stream with configuration:")
    logger.info(f"Input device: {input_device}")
    logger.info(f"Output device: {output_device}")
    logger.info(f"Resolution: {width}x{height}")

    try:
        camera = cv2.VideoCapture(input_device)
        if not camera.isOpened():
            raise RuntimeError("Failed to open camera")
            
        # Get input camera resolution
        input_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
        # Create a small control window
        cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Controls", 400, 100)
        control_frame = np.zeros((100, 400, 3), dtype=np.uint8)
        
        # Initialize virtual camera
        with pyvirtualcam.Camera(width=input_width, height=input_height, fps=30, device=output_device) as virtual_cam:
            print(f'Virtual camera device: {virtual_cam.device}')
            print("Keyboard controls:")
            print("Q - Quit")
            print("T - Toggle head tracking")
            print("8/5 - Eye up/down")
            print("4/6 - Eye left/right")
            print("2 - Reset eye position")
            
            tracker = HeadTracker(smoothing_factor=0.2)
            is_grey = False
            use_head_tracking = True
            eye_left_right = 0.0
            eye_up_down = 0.0
            
            while True:
                ret, frame = camera.read()
                if not ret or frame is None:
                    print("Error: Frame capture failed")
                    continue
                    
                height, width = frame.shape[:2]
                mask_image = None
                metrics = None
                
                if use_head_tracking:
                    try:
                        head_frame, bbox = detect_and_crop_head(frame, tracker)
                    except Exception as e:
                        print(f"Head detection error: {e}")
                        bbox = None
                
                if bbox is not None:
                    x, y, w, h = bbox
                    x, y, w, h = map(int, [x, y, w, h])
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    try:
                        # Extract and process region
                        cropped_region = frame[y:y + h, x:x + w]
                        if cropped_region.size == 0:
                            continue
                            
                        # Store original for diff computation
                        original_crop = cropped_region.copy()
                        
                        
                        if is_grey:
                            gray_cropped = cv2.cvtColor(masked_region, cv2.COLOR_BGR2GRAY)
                            gray_cropped_bgr = cv2.cvtColor(gray_cropped, cv2.COLOR_GRAY2BGR)
                            frame[y:y + h, x:x + w] = gray_cropped_bgr
                            image = Image.fromarray(cv2.cvtColor(gray_cropped_bgr, cv2.COLOR_BGR2RGB))
                        else:
                            image = Image.fromarray(cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB))
                        
                        # Transform and get metrics
                        transformed_image, mask_image, metrics = transform_image(
                            image, model, [eye_left_right, eye_up_down, 0, 0]
                        )
                        
                        # Draw head pose visualization
                        draw_head_pose(frame, metrics['head_euler'], 
                                     (center_x, center_y), scale=100)
                        
                        # Update frame with transformed image
                        transformed_patch = cv2.cvtColor(np.array(transformed_image), 
                                                       cv2.COLOR_RGB2BGR)
                        patch_resized = cv2.resize(transformed_patch, (w, h), 
                                                interpolation=cv2.INTER_NEAREST)
                        patch_resized = np.clip(patch_resized, 0, 255).astype(np.uint8)
                        
                        if (y >= 0 and y + h <= frame.shape[0] and 
                            x >= 0 and x + w <= frame.shape[1]):
                            frame[y:y + h, x:x + w] = patch_resized
                            
                    except Exception as e:
                        print(f"Error processing region: {e}")
                        continue
                
                # Create eye position overlay
                eye_overlay = create_eye_position_overlay(eye_left_right, eye_up_down)
                
                                # Mask out the eyes first
                masked_eyes= mask_eyes(cropped_region)
                #burn=None
                burn=compute_visual_difference_heatmap(original_crop,patch_resized)
                # Create combined display with overlays
                
                display_frame = create_diagnostic_overlay(frame, masked_eyes,eye_overlay,burn ,metrics)
                
                # Convert frame to RGB for virtual camera
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Send to virtual camera
                virtual_cam.send(frame_rgb)
                
                # Update control window with current settings
                control_frame.fill(0)
                cv2.putText(control_frame, f"Tracking: {'ON' if use_head_tracking else 'OFF'}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(control_frame, f"Eye pos: ({eye_left_right:.2f}, {eye_up_down:.2f})", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Controls", control_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    use_head_tracking = not use_head_tracking
                    print(f"Head tracking: {'ON' if use_head_tracking else 'OFF'}")
                elif key in [ord('8'), ord('5'), ord('4'), ord('6'), ord('2')]:
                    step = 0.1
                    if key == ord('8'): 
                        eye_up_down = max(-.50, eye_up_down - step)
                        print(f"Eye up/down: {eye_up_down:.2f}")
                    elif key == ord('5'): 
                        eye_up_down = min(.50, eye_up_down + step)
                        print(f"Eye up/down: {eye_up_down:.2f}")
                    elif key == ord('4'): 
                        eye_left_right = max(-.5, eye_left_right - step)
                        print(f"Eye left/right: {eye_left_right:.2f}")
                    elif key == ord('6'): 
                        eye_left_right = min(.5, eye_left_right + step)
                        print(f"Eye left/right: {eye_left_right:.2f}")
                    elif key == ord('2'): 
                        eye_left_right = 0.0
                        eye_up_down = 0.0
                        print("Eye position reset")
                
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
