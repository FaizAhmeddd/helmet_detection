from ultralytics import YOLO

model = YOLO("yolov8n.pt")

if __name__ == "__main__":
    model.train(data=r"E:\helmet_detection\Helmet Detection.v2i.yolov8\data.yaml", epochs=50, imgsz=640, batch=16, device=[1,0])
