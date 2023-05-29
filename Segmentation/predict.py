from ultralytics import YOLO

model = YOLO("yolov8m_2.pt")

model.predict(source = "2.jpg", show=True, save=True, show_labels=False, show_conf=False, conf=0.5, save_txt=False, save_crop=False)