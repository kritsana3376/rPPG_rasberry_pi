# rPPG_rasberry_pi

หลังจากติดตั้ง rasberry OS
ตรวจสอบ lib
sudo pip3 list
ถ้าไม่มี numpy, scipy,opencv ให้ install 
sudo apt-get install python3-scipy
sudo apt-get install python3-numpy 
sudo apt-get install python3-opencv
ติดตั้ง mediapipe
1. Install FFmpeg and OpenCV from official repository OR Build from sources using this Guide .
sudo apt install ffmpeg python3-opencv python3-pip
2. Install dependency packages
sudo apt install libxcb-shm0 libcdio-paranoia-dev libsdl2-2.0-0 libxv1  libtheora0 libva-drm2 libva-x11-2 libvdpau1 libharfbuzz0b libbluray2 libatlas-base-dev libhdf5-103 libgtk-3-0 libdc1394-22 libopenexr23
สำหรับ Raspberry Pi 4
3. ติดตั้ง  package
sudo pip3 install mediapipe-rpi4
ลบติดตั้ง package
sudo pip3 uninstall mediapipe-rpi4
สำหรับ  Raspberry Pi 3
3. ติดตั้ง  package
sudo pip3 install mediapipe-rpi3
ลบติดตั้ง  package
sudo pip3 uninstall mediapipe-rpi3
