from ultralytics import YOLO

def run_training():
    model = YOLO("yolov8n.pt")  
    model.train(
        data="data.yaml",
        epochs=10,
        imgsz=640,
        batch=16,
        workers=2,  # you can increase later
        device=0
    )

if __name__ == "__main__":   # 👈 required on Windows
    run_training()
