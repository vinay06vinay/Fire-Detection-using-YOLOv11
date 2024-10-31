from ultralytics import YOLO
import argparse
import numpy as np
from typing import List, Dict, Tuple


def get_bbox(results) -> List[Dict[str, float]]:
    """
    Extract bounding boxes, class labels, and coordinates for each detected object in numpy format.
    
    Args:
        results (list): List of prediction results returned by the YOLO model.
    
    Returns:
        List[Dict[str, float]]: A list of dictionaries, each containing:
            - 'class': (int) Class label of the detected object.
            - 'confidence': (float) Confidence score of the detection.
            - 'xyxy': (numpy array) Coordinates of the bounding box in [x_min, y_min, x_max, y_max] format.
            - 'xywh': (numpy array) Center coordinates and size of the bounding box in [x_center, y_center, width, height] format.
    """
    bboxes = []
    if results:
        for result in results:
            for box in result.boxes:
                bbox = {
                    'class': int(box.cls.cpu().item()),
                    'confidence': float(box.conf.cpu().numpy()[0]),
                    'xyxy': box.xyxy.cpu().detach().numpy()[0],  
                    'xywh': box.xywh.cpu().detach().numpy()[0] 
                }
                bboxes.append(bbox)
    return bboxes

def get_centers(bboxes: List[Dict[str, float]]) -> Dict[int, List[Tuple[float, float]]]:
    """
    Calculate the center coordinates of each bounding box, separated by class.
    
    Args:
        bboxes (list): List of bounding box dictionaries returned by get_bbox function.
    
    Returns:
        Dict[int, List[Tuple[float, float]]]: A dictionary where each key is a class label and the value is a list of tuples
              with center coordinates (x_center, y_center).
    """
    box_centers_by_class = {}
    for box in bboxes:
        cls = box['class']
        x_center, y_center = box['xywh'][:2]
        if cls not in box_centers_by_class:
            box_centers_by_class[cls] = []
        box_centers_by_class[cls].append((float(x_center), float(y_center)))
    return box_centers_by_class

def predict(weights: str, source: str, stream: bool, save: bool) -> None:
    """
    Run inference using the YOLO model and print the center coordinates of detected objects by class.
    """
    try:
        # Initialize the YOLO model with the trained weights
        model = YOLO(weights)

        # Infer the model based on trained results
        results = model.predict(
            source,
            save=save,
            stream=stream
        )
        bboxes = get_bbox(results)  # Extract bounding boxes
        box_centers_by_class = get_centers(bboxes)  # Get centers of bounding boxes
        print(box_centers_by_class)  # Print centers grouped by class

    except Exception as e:
        print(f'Error during prediction: {e}')

if __name__ == "__main__":
    # Argument parser for command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/best.pt',help='Path to model checkpoint (Example: "../runs/detect/train/weights/last.pt")')
    parser.add_argument('--source', type=str, default='fire_test.jpg',help='Path to Testing image/ video file. (A URL if its a Youtube video)')
    parser.add_argument('--stream', type=bool, default=False, help='Set True for video files for smoother stream of frames using generator')
    parser.add_argument('--save', type=bool, default=True,help='Save the detections in runs/detect')
    
    args = parser.parse_args()
    args_dict = vars(args)  
    predict(**args_dict)
