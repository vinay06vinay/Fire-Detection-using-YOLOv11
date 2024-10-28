from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")

# Validate the model
metrics = model.val()
print(metrics.box.map)  # map50-95