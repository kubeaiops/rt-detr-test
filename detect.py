import cv2
import numpy as np
from ultralytics import RTDETR
import time
import torch

coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def processImage(model, image, track=True):
    original_size = image.shape[:2] # Get the original size of the image

    # Predict with the model

    # Select the device for computation (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # cpu only will be extremly slow

    # Use the model to predict or track objects in the image
    results = model.track(image, verbose=False, device=device, persist=True, tracker="bytetrack.yaml") if track else model.predict(image, verbose=False, device=device)

    # Process each detection in the results
    for detection in results:
        boxes = detection.boxes.xyxy # Bounding box coordinates
        names = detection.names # Detected object names
        classes = detection.boxes.cls # Class indices
        confs = detection.boxes.conf # Confidence scores
        ids = detection.boxes.id if track else [[None]*len(boxes)]  # Object IDs for tracking

        if ids is None:
            continue

        # Iterate over rows of the tensor
        for box, name, class_idx, conf, id_ in zip(boxes, names, classes, confs, ids):
            xmin, ymin, xmax, ymax = map(int, box) # Convert box coordinates to integers
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 30, 255), 2) # Draw bounding box

            class_name = coco_classes[int(class_idx)]  # Get class name from index
            label = f' ID: {int(id_)} ' if track and id_ is not None else '' # Label with ID if tracking
            label += f'{class_name}: {round(float(conf) * 100, 1)}%' # Add class name and confidence

            # Calculate text size for the label
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
            dim, baseline = text_size[0], text_size[1]

            # Draw rectangle for text background
            cv2.rectangle(image, (int(xmin), int(ymin)), ((int(xmin) + dim[0] // 3) - 20, int(ymin) - dim[1] + baseline), (30, 30, 30), cv2.FILLED)

            # Put text on image
            cv2.putText(image, label, (int(xmin), int(ymin) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image

def traceObjects(model_type='l', output='yes'):

    # Load a model
    if model_type == 'l':
        model = RTDETR('model/custom.pt')  # load an official model
    elif model_type == 'x':    
        model = RTDETR('model/rtdetr-x.pt')  # load an official model
    elif model_type == 'c':    
        model = RTDETR('model/rtdetr-l.pt')  # load a custom model
    else:
        print ('wrong model_type')
        return

    '''
    Overview
    Prioritize accuracy: Choose RTDETR-x if accuracy is paramount, even if it requires slower processing and possibly more resources.
    Require real-time performance: Choose RTDETR-l for real-time applications where speed is crucial.
    Limited resources: Opt for RTDETR-l if hardware resources are constrained.

    Accuracy:
    RTDETR-x: Offers slightly higher accuracy with a reported 54.8% mAP (mean Average Precision) on COCO val2017 dataset.
    RTDETR-l: Shows lower accuracy with 53.0% mAP on the same dataset.
    
    Speed:
    RTDETR-l: Runs faster, achieving 114 FPS on a T4 GPU.
    RTDETR-x: Slower with 74 FPS on the same GPU.
    
    Model Size:
    RTDETR-x: Likely larger and requires more computational resources due to its higher accuracy.
    RTDETR-l: Likely smaller and more efficient due to its faster speed.
    
    Other factors:
    Availability: Both models are readily available within the Ultralytics Python API.
    Ease of use: Both models should be equally easy to use within the API.

    '''

    # Display model information (optional)
    model.info()
    model.overrides['conf'] = 0.3  
    # Set NMS (Non-Maximum Suppression) confidence threshold = 0.3
    # all the detected bounding boxes with confidence scores lower 
    # than a predefined threshold (say, 0.3 in your script) are discarded. This step filters out weak detections.

    model.overrides['iou'] = 0.4  
    # Set NMS Intersection over Union threshold = 0.4
    # if the Intersection over Union between two bounding boxes is greater than 0.4, one of those boxes is considered redundant and will be suppressed.

    model.overrides['agnostic_nms'] = False  # Set whether NMS is class-agnostic
    # Advantage: Class-agnostic NMS can be useful in scenes where objects of different classes often overlap significantly, and you want to ensure that each physical object is only detected once, irrespective of its class.
    # Disadvantage: However, it can lead to issues if distinct objects of different classes are close to each other. It might incorrectly suppress a valid detection.

    model.overrides['max_det'] = 10  # maximum number of detections per image
    #model.overrides['classes'] = [2,3,0] # Define specific classes to detect

    # Initialize video capture with the default camera
    cap = cv2.VideoCapture(0)

    # Get frame width, height, and FPS from the video capture
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change the codec if needed (e.g., 'XVID')
    if output == 'yes':
        out = cv2.VideoWriter('output_video.mp4', fourcc, 15, (frame_width, frame_height))

    # Check if video capture is successfully opened
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    frameId = 0 # Initialize frame counter
    start_time = time.time() # Record start time for FPS calculation
    fps = str() # Initialize FPS string

    while True:
        frameId += 1
        ret, frame = cap.read()
        if not ret:
            break

        frame = processImage(model, frame, track=True)

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if frameId % 10 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps_current = 10 / elapsed_time  # Calculate FPS over the last 20 frames
            fps = f'FPS: {fps_current:.2f}'
            start_time = time.time()  # Reset start_time for the next 20 frames

        cv2.putText(frame, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow("rt-detr", frame)
        if output=='yes':
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture featuresect
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cap.release()
    if output == 'yes':
        out.release()
