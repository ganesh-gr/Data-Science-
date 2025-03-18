import cv2
import math
import argparse

def highlightFace(net, frame, conf_threshold=0.7):  # Function to detect and highlight faces in the frame
    frameOpencvDnn = frame.copy()  # Make a copy of the input frame
    frameHeight = frameOpencvDnn.shape[0]  # Get frame height
    frameWidth = frameOpencvDnn.shape[1]  # Get frame width
    # Convert frame to blob for deep learning model input
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)  # Set the input for the face detection model
    detections = net.forward()  # Perform forward pass to get detections
    faceBoxes = []  # List to store bounding boxes of detected faces
    for i in range(detections.shape[2]):  # Loop through all detections
        confidence = detections[0, 0, i, 2]  # Confidence score for each detection
        if confidence > conf_threshold:  # If confidence is above the threshold
            # Compute the bounding box coordinates
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])  # Append bounding box to the list
            # Draw rectangle around detected face
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes  # Return processed frame and face bounding boxes


# Paths to pre-trained models and configuration files for face, age, and gender detection
faceProto = r"C:\Users\M.Geethasree\OneDrive\Desktop\age & gender prediction\opencv_face_detector.pbtxt"
faceModel = r"C:\Users\M.Geethasree\OneDrive\Desktop\age & gender prediction\opencv_face_detector_uint8.pb"
ageProto = r"C:\Users\M.Geethasree\OneDrive\Desktop\age & gender prediction\age_deploy.prototxt"
ageModel = r"C:\Users\M.Geethasree\OneDrive\Desktop\age & gender prediction\age_net.caffemodel"
genderProto = r"C:\Users\M.Geethasree\OneDrive\Desktop\age & gender prediction\gender_deploy.prototxt"
genderModel = r"C:\Users\M.Geethasree\OneDrive\Desktop\age & gender prediction\gender_net.caffemodel"

# Mean values for pre-processing the input image
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# Age and gender categories
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load the pre-trained models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Use cv2.VideoCapture for video capture
video = cv2.VideoCapture(0)  # Capture video from the webcam; change path for file input
padding = 20  # Padding around detected faces

while cv2.waitKey(1) < 0:  # Main loop to process video frames
    hasFrame, frame = video.read()  # Read a frame from the video
    if not hasFrame:  # If no frame is captured, exit
        cv2.waitKey()
        break

    resultImg, faceBoxes = highlightFace(faceNet, frame)  # Detect faces and highlight them
    if not faceBoxes:  # If no faces are detected
        print("No face detected")

    for faceBox in faceBoxes:  # Loop through each detected face
        # Extract the face region with padding
        face = frame[max(0, faceBox[1] - padding): min(faceBox[3] + padding, frame.shape[0] - 1),
               max(0, faceBox[0] - padding): min(faceBox[2] + padding, frame.shape[1] - 1)]
        
        if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:  # Skip if the face region is invalid
            print("Empty face region, skipping.")
            continue

        # Pre-process the face region and make predictions
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()  # Predict gender
        gender = genderList[genderPreds[0].argmax()]  # Get the label with the highest probability
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds = ageNet.forward()  # Predict age
        age = ageList[agePreds[0].argmax()]  # Get the label with the highest probability
        print(f'Age: {age[1:-1]} years')

        # Annotate the frame with predicted age and gender
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)  # Display the frame with annotations

# Release video resources and close OpenCV windows
video.release()
cv2.destroyAllWindows()
