from ultralytics import YOLO
import argparse

def train(weights, data, epochs, batch_size, resume, plots, save, lr, save_period):
    """
    Trains a YOLO model on a specified dataset.
    """
    # Initialize the YOLO model with the provided weights (pre-trained or custom config)
    model = YOLO(weights)
    
    # Train the model with specified parameters
    results = model.train(
        data=data,
        save=save,
        save_period=save_period,
        epochs=epochs,
        batch=batch_size,
        resume=resume,
        plots=plots,
        lr0=lr
    )


if __name__ == "__main__":
    # Argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="Train a YOLO model with specified parameters.")
    
    # Adding arguments to the parser
    parser.add_argument('--weights', type=str, default='yolo11n.yaml',help='Path to model config or checkpoint (Example: "../runs/detect/train/weights/last.pt")')
    parser.add_argument('--data', type=str, default='fire_config.yaml',help='Path to YAML file specifying dataset paths and class names')
    parser.add_argument('--epochs', type=int, default=10,help='Number of training epochs to execute')
    parser.add_argument('--batch_size', type=int, default=8,help='Batch size for training, adjust based on GPU capability')
    parser.add_argument('--resume', type=bool, default=False,help='Resume training from the last saved checkpoint (True/False)')
    parser.add_argument('--save_period', type=int, default=5,help='Interval (in epochs) for saving checkpoints')
    parser.add_argument('--lr', type=float, default=0.01,help='Initial learning rate for the training process')
    parser.add_argument('--plots', type=bool, default=False, help='Flag to save training/validation plots (True/False)')
    parser.add_argument('--save', type=bool, default=True,help='Flag to save intermediate checkpoints (True/False)')
    
    # Parse arguments and convert to a dictionary
    args = parser.parse_args()
    args_dict = vars(args)  
    
    # Call the train function with the parsed arguments
    train(**args_dict)
