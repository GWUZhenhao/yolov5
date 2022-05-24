# This python script process our original dataset to the YOLO-accepted dataset
import glob
import os, shutil, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# A function to create a folder
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

# Root path for the dataset
dir_dataset = 'D:/GWU/research/dataset/VisDrone2019-DET-train'
dir_images = os.path.join(dir_dataset, 'images')
dir_annotations = os.path.join(dir_dataset, 'annotations')

# Create folders to store new dataset
dir_YOLO_dataset = './dataset'
dir_YOLO_images = os.path.join(dir_YOLO_dataset, 'images')
dir_YOLO_labels = os.path.join(dir_YOLO_dataset, 'labels')
mkdir(dir_YOLO_dataset)
mkdir(dir_YOLO_images)
mkdir(dir_YOLO_labels)

# Process the labels
# Path of the original labels
paths_annotations = glob.glob(os.path.join(dir_annotations, '*'))
for path_annotation in paths_annotations:
    with open(path_annotation, 'r') as f:
        lines = f.readlines()
        name = os.path.basename(path_annotation)
        # Define the file name
        path_original_image = os.path.join(dir_images, '{}.jpg'.format(name[:-4]))
        path_YOLO_image = os.path.join(dir_YOLO_images, '{}.jpg'.format(name[:-4]))
        path_YOLO_label = os.path.join(dir_YOLO_labels, '{}.txt'.format(name[:-4]))
        for line in lines:
            raw = line.split(',')
            if int(raw[4]) == 1:
                # Copy the image
                shutil.copy(path_original_image, path_YOLO_image)

                # Calculate bounding box for the new label
                height, width, channel = cv2.imread(path_original_image).shape
                x1 = int(raw[0])
                y1 = int(raw[1])
                x2 = int(raw[2]) + x1
                y2 = int(raw[3]) + y1
                center_x, center_y = (x1+x2)/2, (y1+y2)/2
                label_index = int(raw[5])
                height_bbox = (y2 - y1)/height
                width_bbox = (x2 - x1)/width
                center_x, center_y = center_x/width, center_y/height

                # Write one line in the new label file
                with open(path_YOLO_label, 'a') as f_YOLO:
                    f_YOLO.write('{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(label_index, center_x, center_y, width_bbox, height_bbox))
                    f_YOLO.write('\n')



