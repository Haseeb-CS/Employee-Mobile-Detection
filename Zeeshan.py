import socket
import pickle
import cv2
import struct
import torch
import os
from ultralytics import YOLO
import time
import uuid
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best2.pt', force_reload=True)

# Function to customize and draw bounding boxes
def customize_bounding_boxes(frame, predictions):
    drowsy_frames = []  # List to store frames with 'Drowsy' detection
    for pred in predictions[0]:
        # Extract bounding box coordinates
        xmin, ymin, xmax, ymax = int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3])
        # Get the class label index
        class_idx = int(pred[5])
        # Define color and line thickness based on class label
        if model.names[class_idx] == 'Drowsy':
            color = (0, 0, 255)  # Red color for 'Drowsy' bounding box
            thickness = 5  # Normal line thickness
            # Draw the customized bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)
            # Add class label to the bounding box
            label = f"{model.names[class_idx]}: {pred[4]:.2f}"  # Example: "Drowsy: 0.90"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            drowsy_frames.append(frame.copy())  # Append the frame to the list
        elif model.names[class_idx] == 'Awake':
            color = (0, 255, 0)  # Green color for 'Awake' bounding box
            thickness = 5  # Bold line thickness
            # Draw the customized bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)
            # Add class label to the bounding box
            label = f"{model.names[class_idx]}: {pred[4]:.2f}"  # Example: "Awake: 0.90"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame, drowsy_frames

# Set up socket
HOST = ''
PORT = 5555
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)

print("Server is listening...")

# Accept client connection
client_socket, client_address = server_socket.accept()
print(f"Connected to {client_address}")

output_dir = 'captured_drowsy_frames'
os.makedirs(output_dir, exist_ok=True)

drowsy_detected = False
drowsy_start_time = None
awake_duration_threshold = 5  # 5 seconds threshold for continuous drowsiness

while True:
    # Receive frame length from client
    frame_length_data = client_socket.recv(struct.calcsize("L"))
    frame_length = struct.unpack("L", frame_length_data)[0]

    # Receive frame data from client
    frame_data = b''
    while len(frame_data) < frame_length:
        packet = client_socket.recv(frame_length - len(frame_data))
        if not packet:
            break
        frame_data += packet

    # Check if frame data is complete
    if len(frame_data) == frame_length:
        # Convert received frame data to frame
        frame = pickle.loads(frame_data)

        # Perform object detection
        results = model(frame)
        frame_with_boxes, drowsy_frames = customize_bounding_boxes(frame.copy(), results.pred)
        cv2.imshow("Drowsiness Detection", frame_with_boxes)

        if any(pred[5] == 0 for pred in results.pred[0]):
            if not drowsy_detected:
                drowsy_detected = True
                drowsy_start_time = time.time()
            else:
                drowsy_duration = time.time() - drowsy_start_time
                if drowsy_duration >= awake_duration_threshold:
                    for drowsy_frame in drowsy_frames:
                        cv2.imwrite(os.path.join(output_dir, f"sleeping_{uuid.uuid4()}.jpg"), drowsy_frame)
                    drowsy_detected = False

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

client_socket.close()
server_socket.close()
cv2.destroyAllWindows()