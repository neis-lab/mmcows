from ultralytics import YOLO
import yaml
import argparse

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Args for training detector')

    # Define the arguments
    parser.add_argument('--data_directory', type=str, required=True, help='Path to the data directory containing all the folds')
    parser.add_argument('--epochs', type=int, required=True, help='Epochs to train')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    
    
    # Parse the arguments
    args = parser.parse_args()  

    # Modify yaml file to enable training multiple folds
    for fold in ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']:
        print(f'\nTraining fold {fold}')
        # Load the YAML file
        yaml_file = 'custom_data.yaml'

        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Modify the yaml file
        for split in ['train', 'val', 'test']:
            data[split] = f'{args.data_directory}/{fold}/{split}/images'

        # Save the changes back to the YAML file
        with open(yaml_file, 'w') as file:
            yaml.dump(data, file)

        # Verify the change:
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
        print(data)    

        # Load a model
        model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)

        # Train the model
        model.train(
            data='custom_data.yaml', 
            epochs=args.epochs, 
            device=[0, 1, 2, 3], 
            imgsz=640,
            plots = True,
            name = f'run_{fold}_',
            batch = args.batch_size
            )


if __name__ == "__main__":
    main()