import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
import cv2
import json
import yaml


def get_celltype_class_map(csv_file_paths):
    """
    Get a dictionary mapping cell types to their class.
    """
    cell_types_count = {}

    for csv_file_path in csv_file_paths:
        df = pd.read_csv(csv_file_path)
        cell_counts = df['raw_classification'].value_counts()
        for cell_type, count in cell_counts.items():
            if cell_type in cell_types_count:
                cell_types_count[cell_type]['count'] += count
            else:
                cell_types_count[cell_type] = {'count': 0, 'id': None}

    # Sort cell types by alphabetical order
    cell_types = sorted(cell_types_count.keys())

    # Assign class id to each cell type
    for i, cell_type in enumerate(cell_types):
        cell_types_count[cell_type]['id'] = i

    return cell_types_count


def csv_to_yolo_format(csv_file, img_file, celltype_class_map, yolo_base_dir, split):
    """
    Convert a csv file to yolo format.
    """
    img = cv2.imread(img_file)
    height, width, _ = img.shape
    
    df = pd.read_csv(csv_file)
    df['class'] = df['raw_classification'].map(celltype_class_map)
    df['x'] = ((df['xmin'] + df['xmax']) / 2) / width
    df['y'] = ((df['ymin'] + df['ymax']) / 2) / height
    df['w'] = (df['xmax'] - df['xmin']) / width
    df['h'] = (df['ymax'] - df['ymin']) / height

    # Clip values to be between 0 and 1
    df[['x', 'y', 'w', 'h']] = df[['x', 'y', 'w', 'h']].clip(lower=0, upper=1)

    # txt file path
    save_dir = os.path.join(yolo_base_dir, 'labels', split)
    txt_file = os.path.basename(csv_file).replace('csv', 'txt')

    df[['class', 'x', 'y', 'w', 'h']].to_csv(
        os.path.join(save_dir, txt_file),
        header=False, index=False, sep=' '
    )

    # copy image to yolo images directory
    img_save_dir = os.path.join(yolo_base_dir, 'images', split)
    img_save_path = os.path.join(img_save_dir, os.path.basename(img_file))
    cv2.imwrite(img_save_path, img)



def create_yolo_config_file(celltype_class_map):
    """
    Create a yolo config file.
    """
    data = {
        'path': '/yolo_workspace/data/',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(celltype_class_map),
        'names': list(celltype_class_map.keys())
    }

    with open('nucls.yaml', 'w') as f:
        yaml.dump(data, f)





def preprocess_data():
    csv_dir = './data/csv'
    img_dir = './data/rgb'

    yolo_base_dir = './data/'

    csv_files = os.listdir(csv_dir)
    img_files = os.listdir(img_dir)

    # Remove csv files that don't have corresponding images and duplicates
    csv_files_filtered = []
    img_files_filtered = []
    for img_file in img_files:
        img_name = img_file.split('.')[0]
        csv_file_corr = img_name + '.csv'
        for csv_file in csv_files:
            if csv_file == csv_file_corr:
                csv_files_filtered.append(csv_file)
                img_files_filtered.append(img_file)
                break

    csv_file_paths = [os.path.join(csv_dir, file) for file in csv_files_filtered]
    img_file_paths = [os.path.join(img_dir, file) for file in img_files_filtered]

    csv_img_pairs = list(zip(csv_file_paths, img_file_paths))

    # Split data into train, test, and validation sets
    train, test = train_test_split(csv_img_pairs, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.2, random_state=42)

    # Get cell type to class mapping
    celltype_class_map = get_celltype_class_map(csv_file_paths)

    tot_count = sum(class_info['count'] for class_info in celltype_class_map.values())
    print(f'Total number of cells: {tot_count}')

    celltype_class_map_filtered = {}
    weights = [0] * len(celltype_class_map)
    for cell_type, class_info in celltype_class_map.items():
        celltype_class_map_filtered[cell_type] = class_info['id']
        weights[class_info['id']] = tot_count / class_info['count']

    print(f'Number of classes: {len(celltype_class_map_filtered)}')
    print(f'Class weights: {weights}')



    # make directories for yolo annotations
    os.makedirs(os.path.join(yolo_base_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(yolo_base_dir, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(yolo_base_dir, 'images', 'val'), exist_ok=True)

    os.makedirs(os.path.join(yolo_base_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(yolo_base_dir, 'labels', 'test'), exist_ok=True)
    os.makedirs(os.path.join(yolo_base_dir, 'labels', 'val'), exist_ok=True)

    # Convert csv files to yolo format
    for csv_file, img_file in train:
        csv_to_yolo_format(csv_file, img_file, celltype_class_map_filtered, yolo_base_dir, 'train')

    for csv_file, img_file in test:
        csv_to_yolo_format(csv_file, img_file, celltype_class_map_filtered,yolo_base_dir, 'test')

    for csv_file, img_file in val:
        csv_to_yolo_format(csv_file, img_file, celltype_class_map_filtered, yolo_base_dir, 'val')


    # Create yolo config file
    create_yolo_config_file(celltype_class_map_filtered)


    # find average width and height of images
    img_sizes = []
    for img_file in img_file_paths:
        img = cv2.imread(img_file)
        height, width, _ = img.shape
        img_sizes.append(width * height)
    
    avg_img_size = np.mean(img_sizes)
    print(f'Average image size: {np.sqrt(avg_img_size)} pixels')

if __name__ == '__main__':
    preprocess_data()


