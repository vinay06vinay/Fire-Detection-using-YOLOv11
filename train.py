from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml") 

# Train the model
results = model.train(data="fire_config.yaml", epochs=10,batch=9)