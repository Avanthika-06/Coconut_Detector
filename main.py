from ultralytics import YOLO
model=YOLO("yolov8n.yaml")
results = model.train(data="data.yaml",epochs=100)

# model = YOLO('./runs/detect/train/weights/last.pt')
# image_path = 'C:/Users/ashwi/Downloads/data/train/images/IMG_0169-1-_JPG.rf.650846a0556b2bf0689508e6610de776.jpg'
# results = model(image_path, save=True, conf=0.3)
# print(results[0].boxes)