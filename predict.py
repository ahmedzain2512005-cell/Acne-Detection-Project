from ultralytics import YOLO
import cv2

# تحميل الموديل YOLO
model = YOLO("best.pt")

# قراءة الكلاسات من داخل الموديل (صح 100%)
classes = list(model.names.values())

def predict(image_path):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("الصورة مش موجودة!")

    results = model(image_path)

    probs = results[0].probs
    class_id = int(probs.top1)

    return classes[class_id]

if __name__ == "__main__":
    try:
        image_path = "images.jpg"   # ← غيّر الصورة فقط
        pred = predict(image_path)
        print("Prediction:", pred)

    except Exception as e:
        print("خطأ:", e)