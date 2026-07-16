from ultralytics import YOLO
import cv2

model = YOLO("yolo26n.pt", task="detect")

results = model.predict(
    source="qwee.jpeg",
    conf=0.20,
    imgsz=640,
    device="cpu"
)

boxed_image = results[0].plot()

cv2.imwrite("3.jpg", boxed_image)