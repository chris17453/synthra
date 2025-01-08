# 
#  ░▒▓███████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░ ░▒▓██████▓▒░  
# ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
# ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
#  ░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓████████▓▒░▒▓███████▓▒░░▒▓████████▓▒░ 
#        ░▒▓█▓▒░  ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
#        ░▒▓█▓▒░  ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
# ░▒▓███████▓▒░   ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
#                                                                                             
# Created By: Charles Watkins                                                                                            
# Date : 2025-01-06
# 

import argparse
import glob
import os
from .model import load_model
from .stream import start_stream

def find_camera_index(device_path):
    """Find the index of a camera given its v4l2 device path"""
    # List all video devices
    video_devices = glob.glob('/dev/video*')
    try:
        # Get the device number from the path (e.g., /dev/video0 -> 0)
        if device_path in video_devices:
            return int(device_path.replace('/dev/video', ''))
        # If it's already a number, return it
        return int(device_path)
    except ValueError:
        raise argparse.ArgumentTypeError(f'Invalid device: {device_path}. Available devices: {", ".join(video_devices)}')

def parse_resolution(res_str):
    try:
        width, height = map(int, res_str.split('x'))
        return width, height
    except ValueError:
        raise argparse.ArgumentTypeError('Resolution must be in format WxH (e.g., 1920x1080)')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Synthra - AI-enabled virtual camera system',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-i', '--input-device',
        help='Input video device (e.g., /dev/video2)',
        default='/dev/video2',
        type=find_camera_index
    )
    
    parser.add_argument(
        '-o', '--output-device',
        help='Output v4l2loopback device (e.g., /dev/video1)',
        default="/dev/video1",
        type=str
    )
    
    parser.add_argument(
        '--input-resolution',
        help='Input resolution in WxH format (e.g., 1920x1080)',
        default='1920x1080',
        type=parse_resolution
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    width, height = args.input_resolution
    model = load_model()
    start_stream(model, 
                input_device=args.input_device,  # Will always be an index
                output_device=args.output_device,
                width=width,
                height=height)