import cv2
import socket
import pickle
import struct
import threading
from datetime import datetime, timedelta
from Facial_Recognition import FacialRecognition
from Mobile_detection import MobileDetection

# Objects of the class
facial_recognition = FacialRecognition()
mobile_detection = MobileDetection()

lock = threading.Lock()

last_mobile_detection_time = None
continuous_detection_start_time = None
frame_saved = False
mobile_detection_count = 0

MOBILE_DETECTION_THRESHOLD = 30
TIME_FRAME_SECONDS = 5

def receive_live_video(client_socket):
    try:
        def facial_recognition_thread(frame):
            facial_recognition.recognize_faces(frame)

        def mobile_detection_thread(frame):
            global last_mobile_detection_time
            global continuous_detection_start_time
            global frame_saved
            global mobile_detection_count

            with lock:
                mobile_detection.detect_mobiles(frame)

                if mobile_detection.mobile_detected:
                    current_time = datetime.now()

                    if last_mobile_detection_time is None or (current_time - last_mobile_detection_time).total_seconds() < TIME_FRAME_SECONDS:
                        last_mobile_detection_time = current_time
                        continuous_detection_start_time = None
                        frame_saved = False
                        mobile_detection_count += 1

                        
                        if mobile_detection_count >= MOBILE_DETECTION_THRESHOLD:
                            threading.Thread(target=save_frame, args=(frame,)).start()
                            mobile_detection_count = 0
                    else:
                        
                        threading.Thread(target=save_frame, args=(frame,)).start()
                        last_mobile_detection_time = current_time
                        continuous_detection_start_time = None
                        frame_saved = False
                        mobile_detection_count = 0

        def save_frame(frame):
            global frame_saved  

            if not frame_saved:
                filename = f"mobile_detection_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                cv2.imwrite(filename, frame)
                print(f"Frame saved: {filename}")
                continuous_detection_start_time = None
                frame_saved = True

        while True:
            data_size = client_socket.recv(struct.calcsize("L"))

            if not data_size:
                print("Connection ended successfully")
                client_socket.close()
                break

            message_size = struct.unpack("L", data_size)[0]

            data = b""
            while len(data) < message_size:
                packet = client_socket.recv(message_size - len(data))
                if not packet:
                    break
                data += packet

            frame = pickle.loads(data)

            thread_facial = threading.Thread(target=facial_recognition_thread, args=(frame,))
            thread_mobile = threading.Thread(target=mobile_detection_thread, args=(frame,))

            thread_facial.start()
            thread_mobile.start()

            thread_facial.join()
            thread_mobile.join()

            cv2.imshow("Received Frame", frame)
            cv2.waitKey(1)
    except Exception as e:
        print("An error occurred:", e)
        client_socket.close()

if __name__ == "__main__":
    SERVER_PORT = 12347

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_socket.bind(("0.0.0.0", SERVER_PORT))
    server_socket.listen(5)
    print("Waiting for incoming connection...")

    while True:
        client_socket, _ = server_socket.accept()
        print("Connected to client.")
        threading.Thread(target=receive_live_video, args=(client_socket,)).start()
