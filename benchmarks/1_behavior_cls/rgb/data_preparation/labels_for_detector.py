import os
import argparse
# Manually labeled classes are 1-16, yolo needs them to be 0-15
# Hence, we subtract one from each

# For training the Cow Detector, we are converting all labels to 0, ie. single class of Cow
def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process annotations with specific splits.')

    # Define the arguments
    parser.add_argument('--dataset_path', type=str, required=True, help='dataset_path_containing_folds')
    # parser.add_argument('--fold', type=str, required=True, help='Fold to prepare labels')

    # Parse the arguments
    args = parser.parse_args()  
    # n_empty_files = 0 
    for fold in ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']:
        print(f'Processing fold {fold}...')
        for subfolder in ['train', 'val', 'test']:
            print(f'Updating labels for split {subfolder}')
            folder_path = os.path.join(args.dataset_path, fold, subfolder, 'labels')
            # Iterate over all files in the folder
            for filename in os.listdir(folder_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, 'r') as file:
                        lines = file.readlines()
                        """
                        # use this code to check for empty label files and delete them
                        if len(lines) == 0:
                            print(f'Empty file: {file_path}')
                            n_empty_files += 1
                            # os.remove(file_path)
                        """

                    with open(file_path, 'w') as file:
                        for line in lines:
                            parts = line.strip().split(' ')
                            if len(parts) > 0:
                                class_label = int(parts[0])
                                updated_label = 0 # class_label - 1
                                updated_line = str(updated_label) + ' ' + ' '.join(parts[1:]) + '\n'
                                file.write(updated_line)
            
            # print('Done')

    # print('\nNumber of empty files: ', n_empty_files, '\n')


if __name__ == "__main__":
    main()