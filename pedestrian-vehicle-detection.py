import torch
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from torchvision import models
# from torchsummary import summary
from torchvision.io import read_image
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import shutil


import os


model = torch.hub.load('yolov5', 'custom', path='yolov5/runs/train/exp/weights/best.pt', source='local')

def make_detection_images(images,model,height=700, width=1000):
    
    # Make detection images for a list of input image paths.

    resized_images = []

    for image in images:
        # Resize the image to the desired width and height.
        resized_image = cv2.resize(image, (width, height))

        # Append the resized image to the list of resized images.
        resized_images.append(resized_image)
    
    results = model(resized_images,size=500)
    # print(results.__dict__)
    print(results.xyxy)
    results.save()


def main():
    st.title("YOLOv5 Object Detection with Streamlit")

    shutil.rmtree('runs', ignore_errors=True)
    # File uploader for image selection
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = Image.open(uploaded_file)
        image = np.array(image)

        # Perform object detection on the uploaded image
        make_detection_images([image], model)

        # Display the detection results
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Detected objects:")

        # show saved image
        # saved_img = cv2.imread('runs/detect/exp/image0.jpg')
        st.image('runs/detect/exp/image0.jpg')

if __name__ == "__main__":
    main()