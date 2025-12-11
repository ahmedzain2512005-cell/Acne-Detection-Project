from ultralytics import YOLO
from roboflow import Roboflow

# ---- 1) تحميل الداتا من Roboflow ----
rf = Roboflow(api_key="XpMwhGeTQsdlI8UuG4OG")
project = rf.workspace("bangkit-academy-rnfpg").project("acne-detection-g5vvz")
version = project.version(1)
dataset = version.download("folder")

data_dir = dataset.location
print("Dataset path:", data_dir)

# ---- 2) تدريب الموديل ----
model = YOLO("yolov8n-cls.pt")

model.train(
    data=data_dir,
    epochs=50,
    imgsz=224,
    batch=32,
    patience=10,
    optimizer="Adam",
    lr0=1e-3,
    val=True,
    verbose=True,
)

# أفضل وزن تم حفظه
best = model.trainer.best
print("Best model saved at:", best)