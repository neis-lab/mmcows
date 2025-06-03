import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from PIL import Image
import json 
from sklearn.metrics import classification_report

# Define the device to use (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Custom_resize_transform(object):
    def __init__(self, output_size = (224, 224)):
        #assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
 
 
    def __call__(self, img):
 
        old_size = img.size # width, height
        ratio = float(self.output_size[0])/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        img = img.resize(new_size)
        # Paste into centre of black padded image
        new_img = Image.new("RGB", (self.output_size[0],self.output_size[1]))
        new_img.paste(img, ((self.output_size[0]-new_size[0])//2, (self.output_size[1]-new_size[1])//2))
        
        return new_img


def test_model(model, dataloader):
    model.eval()  # Set model to evaluate mode
    all_labels = []
    all_preds = []

    # Iterate over data.
    phase_size = len(dataloader)
    with tqdm(total=phase_size, desc='Test Progress', unit='batch') as pbar:
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

            # Collect all labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            pbar.update(1)

    # Generate classification report, ignoring missing classes
    present_classes = list(set(all_labels))
    print('\npresent_classes', present_classes)
    print('\ntarget_names', dataloader.dataset.classes)
    print('Some preds', all_preds[:100])
    report = classification_report(all_labels, all_preds, labels=present_classes, target_names=dataloader.dataset.classes)
    # report = classification_report(all_labels, all_preds, labels=present_classes)
    #report = classification_report(all_labels, all_preds)
    
    print("\nClassification Report:\n")
    print(report)



def create_model(num_classes):
        # Load the pretrained EfficientNetB0 model
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        return model


def check_and_delete_empty_classes(validation_dir):
    """
    Checks for any classes with no samples in the validation split and deletes their empty folders.

    Args:
    validation_dir (str): Path to the validation directory containing subdirectories for each class.

    Returns:
    list: List of class names that were empty and deleted.
    """
    empty_classes = []
    for class_name in os.listdir(validation_dir):
        class_dir = os.path.join(validation_dir, class_name)
        if os.path.isdir(class_dir):
            if not os.listdir(class_dir):  # Check if the directory is empty
                empty_classes.append(class_name)
                os.rmdir(class_dir)  # Delete the empty directory
    return empty_classes


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description='Train an EfficientNetB0 model for image classification')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to the data directory containing test folder')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and evaluation')
    parser.add_argument('--model_path', type=str, required = True, help='Path to save the best model')
    parser.add_argument('--normalization_values_file_path', required=True, type=str, help='Path to json file containing mean and std dev')
    parser.add_argument('--fold', required=True, type=str, help='Fold')
    
    args = parser.parse_args()
    test_dir = args.test_data_path


    norm_params_file = open(args.normalization_values_file_path)
    params = json.load(norm_params_file)
    mean, std = params['behavior'][args.fold]['mean'], params['behavior'][args.fold]['std_dev']
    # Closing file
    norm_params_file.close()


    # remove empty classes from test set
    empty_classes = check_and_delete_empty_classes(test_dir)
    if empty_classes:
        print("The following empty class directories were deleted from the test split:")
        for class_name in empty_classes:
            print(f"- {class_name}")

    # Define transformations with the computed mean and std

    test_transform = transforms.Compose([
                    Custom_resize_transform(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])

    test_dataset = datasets.ImageFolder(test_dir, test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Create model
    model = create_model(num_classes = 7)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)

    # Test the model with the test dataset
    test_model(model, test_dataloader)






