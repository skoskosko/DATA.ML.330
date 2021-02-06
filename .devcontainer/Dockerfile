FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

RUN apt update && apt install -y git libgl1-mesa-glx libglib2.0-0

RUN pip install matplotlib pylint sklearn keras opencv-python mmcv mmdet mmcv-full pandas

# RUN mkdir /home/root

CMD ["/bin/sh", "-c", "while sleep 1000; do :; done"]