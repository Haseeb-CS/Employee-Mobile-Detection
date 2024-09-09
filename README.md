Employee Mobile Detection


This surveillance system enables companies to monitor employees during work hours by detecting mobile phone usage and identifying employees using advanced AI models. It utilizes YOLOv8 for real-time mobile detection and FaceNet for facial recognition, logging incidents on a secure cloud server to ensure accountability and enhance productivity by reducing unauthorized mobile usage in the workplace.


## Authors

- [@Haseeb-CS](https://github.com/Haseeb-CS)


## Features

- Real-time video stream processing from client socket
- Facial recognition to identify employees using the FacialRecognition class
- Mobile detection using the MobileDetection class
- Threaded handling of facial recognition and mobile detection for efficient processing
- Continuous mobile detection tracking within a specified time frame
- Frame saving when mobile detection exceeds the defined threshold
- Cloud-based logging for recorded events
- Configurable parameters for mobile detection threshold and time window
- Error handling for connection and processing failures
- Displays live video frames with real-time analysis results using OpenCV
- Supports multiple client connections with threaded handling


## 🚀 About Me
I'm a Machine Learning Engineer specializing in computer vision, natural language processing (NLP), and image generation. I develop AI solutions that leverage my expertise in these domains to solve complex problems and create innovative applications.


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/Haseeb-CS?tab=repositories)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](www.linkedin.com/in/shahhaseeb281)



## Installation

Follow these steps to set up the surveillance system on your local machine:

Clone the Repository: If you haven't already, clone the repository containing the code:

```
git clone https://github.com/Haseeb-CS/Employee-Mobile-Detection.git

cd Employee-Mobile-Detection
```
Install Python: Make sure Python 3.x is installed on your system. You can download it from the official Python website.

Set Up a Virtual Environment (Optional but Recommended): Create and activate a virtual environment to manage dependencies separately:
```
python -m venv venv
# Activate on Windows
venv\Scripts\activate
# Activate on macOS/Linux
source venv/bin/activate
```
Install Required Packages: Install all the necessary Python packages using pip:
```
pip install opencv-python
pip install ultralytics
pip install numpy
pip install scikit-learn
pip install keras-facenet
pip install pickle5  # Use pickle5 for better compatibility
```
Download YOLOv8 Model:

You may need to download a YOLOv8 pre-trained model for mobile detection. You can do this by initializing the YOLO class in your script or directly from the YOLO repository:
```
yolo = YOLO('yolov8n.pt')  # Replace 'yolov8n.pt' with your specific model version
```
Set Up and Test the Application:

Ensure that your FacialRecognition and MobileDetection classes are correctly defined in their respective modules (Facial_Recognition.py and Mobile_detection.py).
Run the script to start the server:
```
python surveillance_system.py
```
The server will wait for incoming connections from clients to receive and process live video streams.
Troubleshooting:

Make sure all modules are correctly imported and installed.
Verify that your camera or video source is correctly set up and accessible.
If you encounter any issues, check the console for error messages and ensure all dependencies are installed properly.
By following these steps, you should have the surveillance system set up and running smoothly.
    