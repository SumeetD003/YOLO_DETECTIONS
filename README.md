# YOLO_DETECTIONS
This project demonstrates how to read frames from a 10-minute video, detect objects in each frame using the YOLOv3 model, and save the cropped objects into a folder.
## Introduction
YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system. The YOLOv3 version of the model is faster and more accurate compared to previous versions. It divides the image into a grid and predicts bounding boxes and probabilities for each region, enabling the detection of multiple objects in a single frame.

### YOLOv3
YOLOv3 is the third version of the YOLO object detection algorithm. It is an improved version that uses a better feature extractor, known as Darknet-53, which is deeper and more powerful than the ones used in YOLOv1 and YOLOv2. YOLOv3 is capable of detecting objects at different scales using feature maps from different layers of the network.

## Files Used
### yolov3.weights
The yolov3.weights file contains the pre-trained weights for the YOLOv3 model. These weights are obtained by training the YOLOv3 network on the COCO dataset, which contains 80 different classes of objects.

### yolov3.cfg
The yolov3.cfg file contains the configuration details of the YOLOv3 model. It defines the architecture of the network, including the number of layers, the type of layers, the filters used in each layer, and other hyperparameters required for the model.

### coco.names
The coco.names file contains the names of the 80 classes that the YOLOv3 model can detect. Each line in this file corresponds to a class label, such as 'person', 'bicycle', 'car', etc.

## Requirements
To run this project, you need the following Python libraries:

OpenCV
NumPy

## Setup
Download Required Files:

Download yolov3.weights from here.
Download yolov3.cfg from here.
Download coco.names from here.
### Place the Files:

Place yolov3.weights, yolov3.cfg, and coco.names in the same directory as your script.
Ensure your video file is named video.mp4 or update the script to match your video file name.

## Running the Script
The script reads a 10-minute video, detects objects in each frame using YOLOv3, and saves the cropped objects into a folder named cropped_objects.

## Output
The script will save cropped images of detected objects into a folder named cropped_objects.

## Example Outputs
Here are some example snippets of the cropped objects:
![frame_0_object_0](https://github.com/user-attachments/assets/4ba45529-ca8f-459f-990d-be824e1a5805)

![frame_118_object_3](https://github.com/user-attachments/assets/647dc03a-552b-471c-b9be-33f18ec9abbc)

![frame_350_object_5](https://github.com/user-attachments/assets/6760a04d-e2d6-4745-a3a8-e4799c6610a4)


## Video Link
The video used for this project can be found [https://www.youtube.com/watch?v=MCWJNOfJoSM&t=86s.](url)
