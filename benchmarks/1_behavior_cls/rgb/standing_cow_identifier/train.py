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

def compute_mean_std(loader, n_samples = 5000):
    mean = 0.0
    std = 0.0
    total_images_count = 0
    #iteration = 0
    for images, labels in loader:
        #print(iteration)
        #iteration += 1
        #print(images.shape)
        images_count_in_batch = images.size(0)
        images = images.view(images_count_in_batch, images.size(1), -1)
        #print(images.shape)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += images_count_in_batch

        if total_images_count > n_samples:
            break
    mean /= total_images_count
    std /= total_images_count

    return mean, std

def get_mean_std(training_dataset_path, subset_size=5000):
    training_transforms = transforms.Compose([
                Custom_resize_transform(),
                transforms.ToTensor()
                ])

    train_dataset = datasets.ImageFolder(root=training_dataset_path, transform=training_transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    mean, std = compute_mean_std(train_loader)
    print('Marker 2')
    del train_loader
    return mean, std



def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        epoch_start_time = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            phase_size = len(dataloaders[phase])
            with tqdm(total=phase_size, desc=f'{phase.capitalize()} Progress', unit='batch') as pbar:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    pbar.update(1)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
            
                # Ensure the save path exists
                save_dir = args.model_save_path
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # Save the best model weights
                save_path = os.path.join(save_dir, f'best_model_{args.fold}.pt')
                torch.save(best_model_wts, save_path)
                print(f'Saving model to {save_path}')
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        remaining_epochs = num_epochs - (epoch + 1)
        remaining_time = remaining_epochs * epoch_duration

        print(f'Time elapsed: {epoch_duration:.2f}s, Estimated remaining time: {remaining_time/60:.2f}min')

        print()

    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_model(model, dataloaders, criterion):
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    phase_size = len(dataloaders['test'])
    with tqdm(total=phase_size, desc='Test Progress', unit='batch') as pbar:
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            pbar.update(1)

    epoch_loss = running_loss / len(image_datasets['test'])
    epoch_acc = running_corrects.double() / len(image_datasets['test'])

    print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


def create_model(num_classes):
        # Load the pretrained EfficientNetB0 model
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        return model


def check_empty_classes(dataset):
    class_counts = {class_name: 0 for class_name in dataset.classes}
    for _, label in dataset.samples:
        class_name = dataset.classes[label]
        class_counts[class_name] += 1
    
    empty_classes = [class_name for class_name, count in class_counts.items() if count == 0]
    return empty_classes

def filter_empty_classes(dataset, empty_classes):
    indices_to_keep = [i for i, (path, label) in enumerate(dataset.samples) if dataset.classes[label] not in empty_classes]
    dataset = Subset(dataset, indices_to_keep)
    return dataset




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
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory containing train, val, and test folders')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train the model')
    parser.add_argument('--model_save_path', type=str, default = 'saved_models', help='Path to save the best model')
    parser.add_argument('--fold', type=str, required=True, help='fold')
    parser.add_argument('--normalization_values_file_path', required=True, type=str, help='Path to json file containing mean and std dev')

    args = parser.parse_args()

    # Define data directories
    data_dir = os.path.join(args.data_path, args.fold)
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    # mean, std = get_mean_std(train_dir) # calculate on the fly
    # Load precomputed mean and std
    norm_params_file = open(args.normalization_values_file_path)
    params = json.load(norm_params_file)
    mean, std = params['standing'][args.fold]['mean'], params['standing'][args.fold]['std_dev']
    # Closing file
    norm_params_file.close()

    # remove empty classes from val set
    empty_classes = check_and_delete_empty_classes(val_dir)
    if empty_classes:
        print("The following empty class directories were deleted from the validation split:")
        for class_name in empty_classes:
            print(f"- {class_name}")

    # remove empty classes from val set
    empty_classes = check_and_delete_empty_classes(test_dir)
    if empty_classes:
        print("The following empty class directories were deleted from the test split:")
        for class_name in empty_classes:
            print(f"- {class_name}")

    # Define transformations with the computed mean and std
    data_transforms = {
        'train': transforms.Compose([
            Custom_resize_transform(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            Custom_resize_transform(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            Custom_resize_transform(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'val': datasets.ImageFolder(val_dir, data_transforms['val']),
        'test': datasets.ImageFolder(test_dir, data_transforms['test'])
    }

    # Create the dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=args.batch_size, shuffle=False, num_workers=4),
        'test': DataLoader(image_datasets['test'], batch_size=args.batch_size, shuffle=False, num_workers=4)
    }

    # Create model
    model = create_model(num_classes = 16)

    model = model.to(device)
    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate the model
    best_model = train_model(model, dataloaders, criterion, optimizer, num_epochs=args.epochs)


    # Test the model with the test dataset
    test_model(best_model, dataloaders, criterion)






