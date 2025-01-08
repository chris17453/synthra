```bash
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
```
[Synthra](assets/product.jpg)

## preqs
```bash
sudo dnf install python3-opencv v4l2loopback-utils v4l2loopback
pipenv install pillow torchvision torch mediapipe numpy onnxruntime-gpu
```

## Create new virtual cam
- add a config file so the device is persistant
- when changing the kernel module needs to be removed then readded
```bash
# Add to  /etc/modprobe.d/v4l2loopback.conf
# I'm adding 2 because 1 already existed and i dont want to loose it.
# also it seems the first must ALWAYS start with 0?
options v4l2loopback video_nr=0,1 card_label="OBS,Focus Cam" exclusive_caps=1
#OR temp per boot
sudo modprobe v4l2loopback devices=1 video_nr=0 card_label="Custom Camera" exclusive_caps=1
```


## Run Standalone
```bash
cd syntrha
python -m synthra
```

## Run Installed
```bash
synthra
```