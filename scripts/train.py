from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    model.train(
        data='config/dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        project='results',
        save=True
    )