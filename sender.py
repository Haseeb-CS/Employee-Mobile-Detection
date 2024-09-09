import cv2
import socket
import pickle
import struct
import keyboard

def send_live_video(server_address, server_port):
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    video = cv2.VideoCapture(1)

    try:
  
        client_socket.connect((server_address, server_port))

        while True:
    
            success, frame = video.read()
            if not success:
                break

            data = pickle.dumps(frame)

            message_size = struct.pack("L", len(data))
            client_socket.sendall(message_size + data)


            if keyboard.is_pressed("q"):
                break

    except Exception as e:
        print("An error occurred:", e)
    finally:
        # Close the socket and release the camera
        client_socket.close()
        video.release()

if __name__ == "__main__":
    # Server address and port
    SERVER_ADDRESS = "192.168.18.57"  # Change this to your server's IP address or hostname
    SERVER_PORT = 12347            # Change this to your server's port number

    # Call the function to stream live video
    send_live_video(SERVER_ADDRESS, SERVER_PORT)
