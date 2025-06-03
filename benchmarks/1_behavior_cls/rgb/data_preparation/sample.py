import os
import shutil
from datetime import datetime
import json
import argparse


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process annotations with specific splits.')

    # Define the arguments
    parser.add_argument('--data_splits_config_file', type=str, required=True, help='Path to the data splits configuration file.')


    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments (or use them in your application)
    print(f"Data Splits Config File: {args.data_splits_config_file}")


    # Load the configuration file
    with open(args.data_splits_config_file, 'r') as file:
        config = json.load(file)    

    print(config)
    


if __name__ == "__main__":
    main()