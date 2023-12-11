from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO('yolov8x.pt')

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data='/yolo_workspace/nucls.yaml', epochs=100, imgsz=352, batch=16)

