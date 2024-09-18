import ultralytics
import time
import torch
import argparse
import json

def main():

    # Set up device
    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    # Read the JSON config file
    inference_config_file = 'custom_ultralytics_yolov8/inference_config.json'
    with open(inference_config_file, 'r') as json_file:
        inference_config = json.load(json_file)
    yolov8_path = inference_config['cow_detection_model_path']

    model = ultralytics.YOLO(yolov8_path)
    # data_path = '/nfs/oprabhune/MmCows/vision_data/unlabeled_data/0721/images/0721/cam_1'
    # data_path = '/nfs/oprabhune/MmCows/vision_data/detection/organized_data/fold_1/train/images/cam_1_1690271846_02-57-26.jpg'
    #data_path = '/nfs/oprabhune/MmCows/vision_data/detection/organized_data/fold_1/train/images/cam_1_1690272821_03-13-41.jpg'
    # data_path = '/nfs/oprabhune/MmCows/vision_data/detection/organized_data/fold_1/test/images'
    # data_path = '/nfs/oprabhune/MmCows/dummy_data_2/cam_3_1690347416_23-56-56.jpg'
    # data_path = '/nfs/oprabhune/MmCows/vision_data/detection/organized_data/fold_1/test/images/cam_1_1690321016_16-36-56.jpg'
    start_time = time.time()
    results = model.predict(
        args.data_path,
        device = device,
        # show = True,
        # save = True,
        # save_txt  =True,
        # save_crop = True
        )

    print('\nTime taken: ', time.time() - start_time)

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description='Inference Pipeline')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory containing images to be inferred')
    # parser.add_argument('--yolov8_path', type=str, default = '/Users/omkar/Library/CloudStorage/OneDrive-purdue.edu/Omkar_research/CPS_dataset/benchmarking/yolov8_cow_detector.pt')
    args = parser.parse_args()
    main()
