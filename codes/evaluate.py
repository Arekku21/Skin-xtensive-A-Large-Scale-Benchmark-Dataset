# Import required libraries
import pandas as pd
import numpy as np
import os
import torch
import timm
from tqdm import tqdm

import ast

import cv2
import torchvision
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import torchvision
from torch.utils.tensorboard import SummaryWriter
from albumentations.pytorch import ToTensorV2

import tensorboard
import argparse

from torchvision import models

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize

from albumentations import (
    HorizontalFlip, VerticalFlip,  ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    GaussNoise, MotionBlur, MedianBlur, RandomResizedCrop,
    RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize, Rotate,
    ShiftScaleRotate, CenterCrop, Crop, Resize, Rotate, RandomShadow, RandomSizedBBoxSafeCrop,
    ChannelShuffle, MotionBlur, CoarseDropout
)

from transformers import SwinForImageClassification


# argparse for  paths
parser = argparse.ArgumentParser(description='Testing Models on Skin-Xtensive Dataset')

# Parser argument for the type of model
parser.add_argument('--model_type', type=str, default='', help='type of model (default: None)')

# Parser argument for the path to the model state dict
parser.add_argument('--path_saved_state_dict', type=str, default="", help='path to model saved state dict (default: None)')

# Parser argument for the path to save the confusion matrix and heatmaps
parser.add_argument('--save_model_path_folder', type=str, default="", help='path to save folder to save models and logging (default: None)')

parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')

parser.add_argument('--test_csv',type=str,default="",help='path to Test.csv for dataset (default:None)')

# Parse the arguments
args = parser.parse_args()


#! print the inputted variables
print(f"Model Type {args.model_type}")
print(f"Saved dict {args.path_saved_state_dict}\n\n")

# ! Fix the variables
# Declare all variables
img_size = 224
pretrained = True
num_classes = 120
batch_size = args.batch_size

# ! Fix the paths
#important paths
test_csv_path = args.test_csv
dataset_img_path = "./dataset/images" # Path to the folder where images are

# ! Fix the path to save the confusion matrix and heatmaps
Save_model_path = "./" + args.save_model_path_folder

valid_transforms = Compose([
            Resize(img_size, img_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)


# Customised API for my dataset
class CustomDatasetFromImagesForalbumentation(Dataset):
    def __init__(self, csv_path,img_path,transforms):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Read the csv file
        self.data_info = pd.read_csv(csv_path)
        # First column contains the image name
        self.image_arr = np.asarray(self.data_info["image_name"])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info["label"])

        # Third column is for an operation indicator
        self.transforms = transforms
        # Calculate len
        self.data_len = len(self.data_info.index)
        self.img_path = img_path

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Obtain image path
        img_path = os.path.join(self.img_path, single_image_name)
        
        # Open image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Transform image to tensor
        transformed = self.transforms(image=image)
        img_as_tensor = transformed["image"]

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len


# Customised API for my dataset - (binary version)
class CustomDatasetFromImagesForalbumentation_binary(Dataset):
    def __init__(self, csv_path,img_path,transforms):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Read the csv file
        self.data_info = pd.read_csv(csv_path)
        # First column contains the image name
        self.image_arr = np.asarray(self.data_info["image_name"])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info["label"])

        # Third column is for an operation indicator
        self.transforms = transforms
        # Calculate len
        self.data_len = len(self.data_info.index)
        self.img_path = img_path

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Obtain image path
        img_path = os.path.join(self.img_path, single_image_name)
        
        # Open image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Transform image to tensor
        transformed = self.transforms(image=image)
        img_as_tensor = transformed["image"]

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        if single_image_label != 1:

            single_image_label = 0

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len 
    
# Define dataset
test_dataset = CustomDatasetFromImagesForalbumentation(test_csv_path, dataset_img_path,transforms = valid_transforms)
test_dataset_binary = CustomDatasetFromImagesForalbumentation_binary(test_csv_path, dataset_img_path,transforms = valid_transforms)

# Define dataloader
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last = False)
test_loader_binary = DataLoader(test_dataset_binary, batch_size=batch_size, shuffle=False, drop_last = False)

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device = ", device)

# ! Fix the path to pretrained model
# path to pretrained weights
model = None
path_to_pretrained = "./training/" + args.path_saved_state_dict


#! fix the model type

# ! resnet152
if args.model_type == "resnet152":

    model_name = "resnet152"
    model = timm.create_model(model_name, pretrained=False).to(device)

# ! efficientnet
elif args.model_type == "efficientnet":

    model_name = "efficientnet_b5"
    model = timm.create_model(model_name, pretrained=False).to(device)

# ! vgg19
elif args.model_type == "vgg19":
    
    model = models.vgg19(weights=None)

# ! inception_v4
elif args.model_type == "inception_v4":

    model_name = "inception_v4"
    model = timm.create_model(model_name, pretrained=False).to(device)

# ! mobilenetv3_large_150d
elif args.model_type == "mobilenetv3":

    model_name = "mobilenetv3_large_100"
    model = timm.create_model(model_name, pretrained=False).to(device)

# ! densenet121
elif args.model_type == "densenet121":

    model_name = "densenet121"
    model = timm.create_model(model_name, pretrained=False).to(device)

# ! VIT
elif args.model_type == "vit":

    model_name = "vit_base_patch16_224"
    model = timm.create_model(model_name, pretrained=False).to(device)

# ! SwinTransformer
elif args.model_type == "swintrans":

    model_name = "microsoft/swin-base-patch4-window7-224"
    model = SwinForImageClassification.from_pretrained(model_name)

# ! SwinTransformer - timm
elif args.model_type == "timm_swintrans":

    model_name = "swin_base_patch4_window7_224"
    model = timm.create_model(model_name, pretrained=False,num_classes=num_classes).to(device)

else:
    print("Unable to load model")

if args.model_type != "timm_swintrans":
    # Adjust the final layer for the number of classes
    if hasattr(model, 'last_linear'):
        model.last_linear = nn.Linear(model.last_linear.in_features, num_classes)
    elif hasattr(model, 'head'):
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Linear):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif isinstance(model.classifier, nn.Sequential):
            if isinstance(model.classifier[-1], nn.Linear):
                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            else:
                raise ValueError("Unexpected classifier structure in Sequential; please adjust manually.")
        else:
            raise ValueError("Unexpected classifier structure; please adjust manually.")
    else:
        raise ValueError("Unable to locate the classification head for this model.")

# Load your pre-trained model if needed
if pretrained:
    print("Using Pre-Trained Model")
    MODEL_PATH = path_to_pretrained
    model.load_state_dict(torch.load(MODEL_PATH),strict=True)

model.to(device)

# Set the model to evaluation mode
model.eval()

# Initialize a variable to store the number of correct predictions
correct = 0
true_labels = []
pred_labels = []

#auc calculations
probs = []
true_labels_auc = []

# Loop through the test_dataset (multiclass)
with torch.no_grad():
    for images, labels in test_loader:
        # Move the images and labels to the same device as the model
        images = images.to(device)
        labels = labels.to(device)

        # Get the model outputs
        outputs = model(images)

        if args.model_type == "swintrans":
            outputs = outputs["logits"]

        # Get the predicted classes
        _, preds = torch.max(outputs, 1)

        # Compare the predicted classes with the true labels
        correct += torch.sum(preds == labels).item()

        # print("True",labels,"Predict",preds)

        true_append = labels.tolist()
        pred_append = preds.tolist()

        true_labels.append(true_append)
        pred_labels.append(pred_append)

        #auc calculations
        # Apply softmax to get probabilities

        #probability calc for binary
        # probabilities = torch.softmax(outputs, dim=1)[:, 1]

        #probability calc for multiclass
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        
        probs.append(probabilities)
        true_labels_auc.append(labels.cpu().numpy())

# Print the number of correct predictions
print(f"Number of correct predictions (multiclass): {correct} out of {len(test_dataset)}")

true_labels_list = []
pred_labels_list = []

for values in true_labels:
  for vlabels in values:
    true_labels_list.append(vlabels)

for values in pred_labels:
  for vlabels in values:
    pred_labels_list.append(vlabels)

# # Generate a classification report showing overall metrics for binary classification
# var_classification_report = classification_report(true_labels_list, pred_labels_list)
# print(var_classification_report)

# Calculate precision, recall, and F1 score
precision = precision_score(true_labels_list, pred_labels_list,average='weighted')
recall = recall_score(true_labels_list, pred_labels_list,average='weighted')
f1 = f1_score(true_labels_list, pred_labels_list,average='weighted')

# Print the calculated scores
print(f"Precision (Weighted): {precision:.4f}")
print(f"Recall (Weighted): {recall:.4f}")
print(f"F1 Score (Weighted): {f1:.4f}")

######### BINARY VERSION #############
# Initialize a variable to store the number of correct predictions
correct = 0
true_labels = []
pred_labels = []

#auc calculations
probs = []
true_labels_auc = []

# Loop through the test_dataset (binary)
with torch.no_grad():
    for images, labels in test_loader_binary:
        # Move the images and labels to the same device as the model
        images = images.to(device)
        labels = labels.to(device)

        # Get the model outputs
        outputs = model(images)

        if args.model_type == "swintrans":
            outputs = outputs["logits"]

        # Get the predicted classes
        _, preds = torch.max(outputs, 1)

        # Convert predictions and labels to binary (1 if class is 1, else 0)
        binary_preds = (preds == 1).int()
        binary_labels = (labels == 1).int()

        # Compare the predicted classes with the true labels
        correct += torch.sum(binary_preds == labels).item()

        # print("True",labels,"Predict",preds)

        true_append = labels.tolist()
        pred_append = binary_preds.tolist()

        true_labels.append(true_append)
        pred_labels.append(pred_append)

# Print the number of correct predictions
print(f"Number of correct predictions: {correct} out of {len(test_dataset)}")
true_labels_list = []
pred_labels_list = []

for values in true_labels:
  for vlabels in values:
    true_labels_list.append(vlabels)

for values in pred_labels:
  for vlabels in values:
    pred_labels_list.append(vlabels)

# Create a confusion matrix
cm = confusion_matrix(true_labels_list, pred_labels_list)

# Plot the confusion matrix using seaborn heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
# plt.show()

# ! adjust the file path
# Save the plot to a file instead of showing it
plt.savefig(Save_model_path + '/confusion_matrix_binary.png')
plt.close()

# Generate a classification report showing overall metrics for binary classification
var_classification_report = classification_report(true_labels_list, pred_labels_list)
print(var_classification_report)

# Calculate precision, recall, and F1 score for binary classification
precision_binary = precision_score(true_labels_list, pred_labels_list, average='binary')
recall_binary = recall_score(true_labels_list, pred_labels_list, average='binary')
f1_binary = f1_score(true_labels_list, pred_labels_list, average='binary')

# Print precision, recall, and F1 score for binary classification
print("Precision (Binary):", precision_binary)
print("Recall (Binary):", recall_binary)
print("F1 Score (Binary):", f1_binary)


####### calculation for the fitz, exposure and brightness calc ########
# Customised API
class CustomDatasetFromImagesForalbumentation_analytics(Dataset):
    def __init__(self, csv_path,img_path,transforms):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Read the csv file
        self.data_info = pd.read_csv(csv_path)
        # image name
        self.image_arr = np.asarray(self.data_info["image_name"])
        # labels
        self.label_arr = np.asarray(self.data_info["label"])

        #fitz column
        self.fitz_arr = np.asarray(self.data_info["fitzpatrick_label"])

        #exposure
        self.expo_arr = np.asarray(self.data_info["exposure"])

        #sharpness
        self.sharp_arr = np.asarray(self.data_info["sharpness"])

        # Third column is for an operation indicator
        self.transforms = transforms

        # Calculate len
        self.data_len = len(self.data_info.index)
        self.img_path = img_path

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Obtain image path
        img_path = os.path.join(self.img_path, single_image_name)
        
        # Open image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Transform image to tensor
        transformed = self.transforms(image=image)
        img_as_tensor = transformed["image"]

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        #get the fitz,exposure,sharpness
        fizt_label = self.fitz_arr[index]
        expo = self.expo_arr[index]
        sharp = self.sharp_arr[index]

        return (img_as_tensor, single_image_label,fizt_label,expo,sharp)

    def __len__(self):
        return self.data_len


#Define dataset (images) path
test_dataset_analytics = CustomDatasetFromImagesForalbumentation_analytics(test_csv_path, dataset_img_path,transforms = valid_transforms)

# Define dataloader
test_loader_analytics = DataLoader(test_dataset_analytics, batch_size=batch_size, shuffle=False, drop_last = False)

head_category_classes = [1,3,82,26,17,14,57,36,34,2,31,6,37,42,59,64,40,48,20,0,33,52,32,28,49,46,63,13,39,110,45,30,84,5,90,119,47,55,68,65,85]

#storing the data
long_tail_category_dict = {
    "head" : 0,
    "tail" : 0
}

#storing the data
correct_predictions_by_fitz = {
    -1 : 0,
    1 : 0,
    2 : 0,
    3 : 0,
    4 : 0,
    5 : 0,
    6 : 0,
}

correct_prediction_by_expos_simple ={
    "Very Dark" :0, # 0 - 50 
    "Dark" :0, # 51-100
    "Moderate" :0, # 101 - 150 
    "Light" :0, # 151 - 200 
    "Very Light" :0 # 201 - 255 
}

correct_prediction_by_sharpness ={
    "Very Blurry" :0, # 0 - 1,000 
    "Blurry" :0, # 1,001 - 3,000 
    "Slightly Blurry" :0, # 3,001 - 6,000 
    "Moderate" :0, # 6,001 - 10,000 
    "Sharp" :0, # 15,001 - 25,000 
    "Very Sharp" :0, # 15,001 - 25,000 
    "Excessively Sharp" :0 # 25,001 - 40,000+ 
}

# Loop through the test_dataset (multiclass)
with torch.no_grad():
    for images, labels, fitz_labels, expos, sharps in test_loader_analytics:
        # Move the images and labels to the same device as the model
        images = images.to(device)
        labels = labels.to(device)

        # Get the model outputs
        outputs = model(images)

        if args.model_type == "swintrans":
            outputs = outputs["logits"]

        # Get the predicted classes
        _, preds = torch.max(outputs, 1)

        # Compare predicted classes with true labels
        correct_mask = preds == labels

        # Iterate through the batch
        for i in range(len(images)):
            if correct_mask[i]:  # If prediction is correct

                #long tail distribution check
                long_tail_cat = labels[i].item()

                if long_tail_cat in head_category_classes:
                    long_tail_category_dict["head"] += 1
                else:
                    long_tail_category_dict["tail"] += 1

                #fitzlabel checking
                fitz_label = fitz_labels[i].item()
                print(f"Correct prediction for Fitzpatrick label {fitz_label}")

                # Add to the dictionary
                if fitz_label in correct_predictions_by_fitz:
                    correct_predictions_by_fitz[fitz_label] +=1

                #exposure
                expo_label = expos[i].item()
                print(f"Correct prediction for Exposure label {expo_label}")

                # !expo but diff categorising
                if expo_label <= 50:
                    
                    correct_prediction_by_expos_simple["Very Dark"] += 1
                
                elif expo_label <= 100:

                    correct_prediction_by_expos_simple["Dark"] += 1

                elif expo_label <= 150:

                    correct_prediction_by_expos_simple["Moderate"] += 1

                elif expo_label <= 200:

                    correct_prediction_by_expos_simple["Light"] += 1

                elif expo_label > 200:

                    correct_prediction_by_expos_simple["Very Light"] += 1


                #!sharpness score
                sharp_label = sharps[i].item()
                print(f"Correct prediction for Sharpness label {sharp_label}")

                #add to dictionary based on range
                if sharp_label <= 1000.00:
                    
                    correct_prediction_by_sharpness["Very Blurry"] += 1
                
                elif sharp_label <= 3000.00:

                    correct_prediction_by_sharpness["Blurry"] += 1

                elif sharp_label <= 6000.00:

                    correct_prediction_by_sharpness["Slightly Blurry"] += 1

                elif sharp_label <= 10000.00:

                    correct_prediction_by_sharpness["Moderate"] += 1

                elif sharp_label <= 15000.00:

                    correct_prediction_by_sharpness["Sharp"] += 1

                elif sharp_label <= 25000.00:

                    correct_prediction_by_sharpness["Very Sharp"] += 1

                elif sharp_label > 25000.00:

                    correct_prediction_by_sharpness["Excessively Sharp"] += 1

print(correct_predictions_by_fitz)

print(correct_prediction_by_expos_simple)

print(correct_prediction_by_sharpness)

print(long_tail_category_dict)