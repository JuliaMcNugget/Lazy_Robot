import cv2 as cv
import PIL as Image
import numpy as np
import os, time, uuid, pickle

current_id = 0
label_ids = {}
y_labels = []
x_train = []
info = []
path_img = ""

# Set up ORB
orb = cv.ORB_create(nfeatures=500)

# Begin Training
    # Starts a walk through the folders and imagese
for root, dirs, files in os.walk(r'.\Training'):
    for files in files:
    
            # if the files end with any of the following
        if files.endswith("png") or files.endswith("jpg") or files.endswith("jfif"):
                # The path is updated
            path_img = os.path.join(root, files)
                # The label is created
            label = os.path.basename(root).replace(" ", "_").lower()
                # prints the gathered information of the files
            print(os.path.basename(root))
            print(label, path_img)
                # Convert Image to black and white
            img = cv.imread(path_img, cv.IMREAD_GRAYSCALE)
                # detects and calculates the points
            keypoints, descriptors = orb.detectAndCompute(img, None)
            if descriptors is not None:
                if label not in label_ids:
                    label_ids[label] = []
                label_ids[label].append(descriptors)
            if img is None:
                continue
            cv.imshow('ORB Keypoints', img) 
            cv.waitKey(0)


cv.destroyAllWindows()

with open("labels.pkl", "wb") as f:
    pickle.dump(label_ids, f)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")