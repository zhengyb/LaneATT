# Use the NVIDIA TensorFlow image as the base
FROM nvcr.io/nvidia/pytorch:23.02-py3

# Set environment variables to configure tzdata non-interactively
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set the working directory in the container
WORKDIR /app

COPY ./ /app/

RUN apt-get update && apt-get install -y \
            build-essential \
            cmake \
            git \
            ninja-build \
            libglib2.0-0 \
            libsm6 \
            libxext6 \
            libxrender-dev \
            tzdata

# 卸载旧版本
#RUN pip uninstall opencv opencv-python -y
#RUN pip install opencv-python-headless==4.5.5.64
RUN pip install -r requirements.txt
#RUN pip install nvidia-dali-cuda120
# for deployment
RUN pip install onnxmltools
RUN pip install pycuda
#RUN pip install addict
RUN pip uninstall tensorboard -y
RUN pip install tensorboard==2.14.0

RUN cd lib/nms; \
    python setup.py install; \
    cd -

CMD ["bash"]





