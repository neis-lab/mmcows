# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
import os 
import copy

# Define the device to use (GPU if available)
if torch.cuda.is_available():
    device = torch.device("cuda")

elif torch.backends.mps.is_available():
    device = torch.device("mps")

else:
    device = torch.device("cpu")



inference_config_file = 'custom_ultralytics_yolov8/inference_config.json'


# Read the JSON config file
with open(inference_config_file, 'r') as json_file:
    inference_config = json.load(json_file)

# Set variables from the JSON data
fold = inference_config['fold']
behav_model_path = inference_config['behav_model_path']
standing_ID_model_path = inference_config['standing_ID_model_path']
lying_ID_model_path = inference_config['lying_ID_model_path']
save_preds_directory = inference_config['save_preds_directory']

mean_behav = inference_config['norm_params']['behavior'][fold]['mean']
std_behav = inference_config['norm_params']['behavior'][fold]['std_dev']

mean_standing_ID = inference_config['norm_params']['standing'][fold]['mean']
std_standing_ID = inference_config['norm_params']['standing'][fold]['std_dev']

mean_lying_ID = inference_config['norm_params']['lying'][fold]['mean']
std_lying_ID = inference_config['norm_params']['lying'][fold]['std_dev']

n_id_classes = inference_config['n_id_classes']
n_behav_classes = inference_config['n_behav_classes']

# Map model prediction idx to real class names
# For identification
# List of class names (labels from 1 to 16)
id_class_names = [str(i) for i in range(1, n_id_classes+1)]
# Sort the class names alphabetically
class_names_sorted = sorted(id_class_names)
# Create the index-to-class mapping
id_idx_to_class = {idx: class_name for idx, class_name in enumerate(class_names_sorted)}

# For behavior classification
# List of class names (labels from 1 to 16)
behav_class_names = [str(i) for i in range(1, n_behav_classes+1)]
# Sort the class names alphabetically
class_names_sorted = sorted(behav_class_names)
# Create the index-to-class mapping
behav_idx_to_class = {idx: class_name for idx, class_name in enumerate(class_names_sorted)}

class Custom_resize_transform(object):
    def __init__(self, output_size = (224, 224)):
        #assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
 
 
    def __call__(self, img):
 
        old_size = img.size # width, height
        # print(type(old_size), old_size)
        ratio = float(self.output_size[0])/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        img = img.resize(new_size)
        # Paste into centre of black padded image
        new_img = Image.new("RGB", (self.output_size[0],self.output_size[1]))
        new_img.paste(img, ((self.output_size[0]-new_size[0])//2, (self.output_size[1]-new_size[1])//2))
        
        return new_img

def create_preprocess(mean, std):
    # Define preprocessing transformation for the classifier
    preprocess = transforms.Compose([
        Custom_resize_transform(),  # Resize to fit the input size of EfficientNet-B0
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return preprocess

def remove_duplicate_bboxes(predictions):
    # Convert predictions to numpy array for easier processing
    predictions = np.array(predictions)
    # Initialize a dictionary to keep the highest confidence bbox per class
    highest_confidence_bboxes = {}

    for prediction in predictions:
        class_label = int(prediction[0])
        confidence = prediction[-1]
        behavior = prediction[-2]
        

        if class_label not in highest_confidence_bboxes:
            highest_confidence_bboxes[class_label] = prediction
        
        elif behavior != 7 and highest_confidence_bboxes[class_label][-2] == 7:
            highest_confidence_bboxes[class_label] = prediction

        elif behavior != 7 and highest_confidence_bboxes[class_label][-2] != 7 and confidence > highest_confidence_bboxes[class_label][-1]:
            highest_confidence_bboxes[class_label] = prediction
        
        elif behavior == 7 and highest_confidence_bboxes[class_label][-2] == 7 and confidence > highest_confidence_bboxes[class_label][-1]:
            highest_confidence_bboxes[class_label] = prediction

    # Extract the filtered bounding boxes
    filtered_bboxes = list(highest_confidence_bboxes.values())
    # Remove the last element (confidence value) of each sublist
    filtered_bboxes = [sublist[:-1] for sublist in filtered_bboxes]
    return filtered_bboxes

"""
def create_UNM_model(num_classes):
    model_ft = models.efficientnet_b0()
    # Get the length of class_names (one output unit for each class)

    # Recreate the classifier layer and seed it to the target device
    model_ft.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=1280, 
                        out_features=num_classes, # same number of output units as our number of classes
                        bias=True))

    #num_ftrs = model_ft.fc.in_features
    num_ftrs = 1280

    # Change 4) Make sure the final dense layer has #neurons = #classes
    # model_ft.fc = nn.Linear(num_ftrs, 16) # For Cow_id classification
    model_ft.fc = nn.Linear(num_ftrs, num_classes) # For Behavior Classification
    return model_ft

"""
def create_OMK_model(num_classes):
        # Load the pretrained EfficientNetB0 model
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        model.eval()
        return model

"""
# load behav model
state_dict = torch.load(behav_model_path)
new_state_dict = {}
for k, v in state_dict.items():
    k = k.replace("module.", "")
    new_state_dict[k] = v
    state_dict = new_state_dict
behavior_classifier = create_UNM_model(num_classes=7).to(device)
behavior_classifier.load_state_dict(new_state_dict)


# load standing ID model
state_dict = torch.load(standing_ID_model_path)
new_state_dict = {}
for k, v in state_dict.items():
    k = k.replace("module.", "")
    new_state_dict[k] = v
    state_dict = new_state_dict
standing_ID_classifier = create_UNM_model(num_classes = 16).to(device)
standing_ID_classifier.load_state_dict(new_state_dict)
"""

# Create and load behav ID model
behavior_classifier = create_OMK_model(num_classes = n_behav_classes).to(device)
behavior_classifier.load_state_dict(torch.load(behav_model_path, map_location=device))
behavior_classifier.eval()

# Create and load lying model
lying_ID_classifier = create_OMK_model(num_classes = n_id_classes).to(device)
lying_ID_classifier.load_state_dict(torch.load(lying_ID_model_path, map_location=device))
lying_ID_classifier.eval()

# Create and load standing ID model
standing_ID_classifier = create_OMK_model(num_classes = 16).to(device)
standing_ID_classifier.load_state_dict(torch.load(standing_ID_model_path, map_location=device))
standing_ID_classifier.eval()

# preprocess transforms
behav_preprocess = create_preprocess(mean_behav, mean_behav)
standing_ID_preprocess = create_preprocess(mean_standing_ID, std_standing_ID)
lying_ID_preprocess = create_preprocess(mean_lying_ID, std_lying_ID)

# Create the directory if it does not exist
os.makedirs(save_preds_directory, exist_ok=True)
for sub_dir in ['cam_1', 'cam_2', 'cam_3', 'cam_4']:
    os.makedirs(f"{save_preds_directory}/{sub_dir}", exist_ok=True)


def behav_and_id(results, orig_img, filename):
        # Process each detected object
        for result in results:
            custom_result_list = []
            for index, obj in enumerate(result.boxes):
                custom_object = []
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, obj.xyxy[0].cpu().numpy())
                # Crop the object from the image
                orig_img_PIL = Image.fromarray(orig_img)
                cropped_obj = orig_img_PIL.crop((x1, y1, x2, y2))
                # cropped_obj = orig_img[y1:y2, x1:x2]
                # cropped_obj.save(f"cropped_im_{index}.jpeg")

                if cropped_obj.size[0] > 0 and cropped_obj.size[1] > 0:  # Ensure cropped image is valid
                    # Preprocess the cropped object for behav recog
                    
                    input_tensor_behav = behav_preprocess(copy.deepcopy(cropped_obj)).unsqueeze(0).to(device) # Add batch dimension

                    # Run the custom classifier on the preprocessed object
                    with torch.no_grad():
                        behavior_classifier_output = behavior_classifier(input_tensor_behav)

                    # Get the predicted class
                    _, predicted_behavior = torch.max(behavior_classifier_output, 1)
                    predicted_behavior_idx = int(predicted_behavior.item()) 
                    # print(f"Predicted behavior: {predicted_behavior}")
                    predicted_behavior = int(behav_idx_to_class[predicted_behavior_idx])
                    
                    if(predicted_behavior == 7):
                        input_tensor_ID = lying_ID_preprocess(copy.deepcopy(cropped_obj)).unsqueeze(0).to(device) # Add batch dimension
                        with torch.no_grad():
                            ID_classifier_output = lying_ID_classifier(input_tensor_ID)   

                    else:
                        input_tensor_ID = standing_ID_preprocess(copy.deepcopy(cropped_obj)).unsqueeze(0).to(device) # Add batch dimension
                        with torch.no_grad():
                            ID_classifier_output = standing_ID_classifier(input_tensor_ID)   
                    
                    probabilities = F.softmax(ID_classifier_output, dim=1)
                    # Get the predicted class and its confidence score
                    confidence, predicted_ID_idx = torch.max(probabilities, 1)
                    predicted_ID_idx = int(predicted_ID_idx.item())
                    predicted_ID = int(id_idx_to_class[predicted_ID_idx])
                    ID_confidence = confidence.item()
                    # print(f"Predicted ID: {predicted_ID}")                      
                    
                    custom_object.append(predicted_ID)
                    custom_object.extend(obj.xywhn.tolist()[0])
                    custom_object.append(predicted_behavior)
                    custom_object.append(ID_confidence)
                    
                    custom_result_list.append(custom_object)
                    """
                    plt.figure(figsize=(3, 3))  # Set figure size to be smaller
                    plt.imshow(cropped_obj)
                    plt.title(f"Predicted behavior: {predicted_behavior}, Predicted ID: {predicted_ID}", fontsize=9)  # Set title font size
                    plt.axis('off')  # Remove x and y axis scales and labels
                    plt.show()
                    """
                else:
                    print("Invalid crop detected, skipping...")
            
            # print('Results 2 : ' , custom_result_list)
            custom_result_list = remove_duplicate_bboxes(custom_result_list)
            # print('Results 2 : ' , custom_result_list)
            text_filename = os.path.splitext(os.path.basename(filename))[0]
            cam_directory = text_filename[0:5]
            text_filename = text_filename[6:]
            # Open the file in write mode
            with open(f'{save_preds_directory}/{cam_directory}/{text_filename}.txt', 'w') as file:
                for sublist in custom_result_list:
                    # Join the elements of the sublist into a space-separated string
                    line = ' '.join(map(str, sublist))
                    # Write the line to the file followed by a newline character
                    file.write(line + '\n')
                # print(f"\nCustom list saved at {text_filename}.txt")

class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model='yolov8n.pt', source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """






    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        # print('len(preds)', len(preds))
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            # print('\n', type(orig_img), orig_img.shape)
            # plt.imshow(orig_img)
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            # print('After Pred', type(pred), pred)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
            # print('Results : ', results)
            behav_and_id(results, orig_img, img_path)
        
        
        return results
