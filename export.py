from ultralytics import YOLO
 
model = YOLO("yolov8n.pt")  # загрузите предварительно обученную модель YOLOv8n
model.export(format="onnx")  