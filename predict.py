from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")

# Predict with the model
results = model("fire_test2.jpg",save=True)  # predict on an image

# Process results list
for result in results:
    print(result.boxes)
