
# Use the official CUDA and cuDNN base image
FROM ultralytics/ultralytics:latest

# Set the working directory
WORKDIR /yolo_workspace

# Install dependencies
RUN apt-get update && apt-get install wget

# get yolov8 weights
RUN wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Copy your training data to the container (replace <path_to_data> with the actual path)
COPY ./ /yolo_workspace


