from ultralytics import YOLO
import yaml
import argparse

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Args for testing detector')

    # Define the arguments
    parser.add_argument('--data_directory', type=str, required=True, help='Path to the data directory containing all the folds')
    # Parse the arguments
    args = parser.parse_args()
    # Modify data.yaml file to enable training multiple folds
    for fold in ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']:
        print(f'\nTesting fold {fold}')
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
        
        # Load the trained model
        model = YOLO(f"runs/detect/run_{fold}_/weights/best.pt")  # load the model

        # Test the model
        metrics = model.val(
            data='custom_data.yaml', 
            device=[0, 1, 2, 3], 
            split = 'test', 
            imgsz=640
            )
        metrics.box.map  # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps  # a list contains map50-95 of each category
        print('\n\n')   

if __name__ == "__main__":
    main()        