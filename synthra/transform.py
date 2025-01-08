import cv2
import torch
import torchvision.transforms as transforms
import numpy as np

def quaternion_to_euler(q):
    w, x, y, z = q
    
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)
    
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw]) * 180.0 / np.pi


def transform_image(image, ort_session, head_gaze_delta):
    preprocess = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.ToTensor(),
    ])

    input_tensor = preprocess(image).unsqueeze(0).numpy().astype(np.float16)
    
    inputs = {
        "src": input_tensor,
        "head_gaze_delta": head_gaze_delta,
    }

    outputs = ort_session.run(None, inputs)
    
    generated_image = outputs[0]
    w_d = outputs[1]
    w = outputs[2]
    s_tx_ty = outputs[3]
    conf = outputs[4]

    output_tensor = torch.tensor(generated_image).squeeze(0)
    rgb_tensor = output_tensor[:3, :, :]
    
    postprocess = transforms.ToPILImage()
    transformed_image = postprocess(rgb_tensor)
    
    mask_tensor = output_tensor[2:3, :, :]
    mask_image = postprocess(mask_tensor.repeat(3, 1, 1))
    
    metrics = {
        'confidence': float(conf[0][0]),
        'gaze_direction': s_tx_ty[0],
        'head_quaternion': w[0],
        'head_euler': quaternion_to_euler(w[0]) * np.pi / 180.0,
        'delta_quaternion': w_d[0]
    }
    
    return transformed_image, mask_image, metrics
