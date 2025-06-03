import ultralytics
import time
import torch
import argparse
import json
import gc

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

    # Load YOLO model
    model = ultralytics.YOLO(yolov8_path)

    # Start timing
    start_time = time.time()

    # Initialize counters
    batch_size = 64  # This will change for each machine based on its memory capacity
    total_images_inferred = 0
    batch_start_time = time.time()

    # Stream predictions in batches
    batch = []
    for img in model.predict(args.data_path, device=device, stream=True, save_txt=True):
        batch.append(img)
        
        # If batch is full, process it
        if len(batch) == batch_size:
            print(f"Processing batch of {len(batch)} images...")

            # Simulate processing (inference happens when streaming)
            total_images_inferred += len(batch)
            
            # Log timing for the current batch
            batch_end_time = time.time()
            print(f"Batch completed in {batch_end_time - batch_start_time:.2f} seconds")
            print(f"Total images inferred: {total_images_inferred}")

            # Clear memory
            batch.clear()
            batch_start_time = time.time()  # Reset batch timer

            # Explicitly release memory
            torch.cuda.empty_cache()  # For CUDA
            gc.collect()  # Ensure Python objects are freed

    # Process any remaining images in the last batch
    if batch:
        print(f"Processing final batch of {len(batch)} images...")
        total_images_inferred += len(batch)
        batch.clear()
        torch.cuda.empty_cache()
        gc.collect()

    # Log total time taken
    print(f"\nTotal time taken: {time.time() - start_time:.2f} seconds")
    print(f"Total images inferred: {total_images_inferred}")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description='Inference Pipeline')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory containing images to be inferred')
    args = parser.parse_args()
    main()
