import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras_facenet import FaceNet
import pickle

class FacialRecognition:
    def __init__(self):
        self.facenet = FaceNet()
        self.faces_embeddings = np.load("faces_embeddings_done_part2_3classes.npz")
        self.Y = self.faces_embeddings['arr_1']
        self.encoder = LabelEncoder()
        self.encoder.fit(self.Y)
        self.haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

    def recognize_faces(self, frame):
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haarcascade.detectMultiScale(gray_img, 1.3, 5)

        for (x, y, w, h) in faces:
            img = rgb_img[y:y + h, x:x + w]
            img = cv2.resize(img, (160, 160))
            img = np.expand_dims(img, axis=0)
            ypred = self.facenet.embeddings(img)
            face_name = self.model.predict(ypred)

            probabilities = self.model.predict_proba(ypred)
            max_probability = np.max(probabilities)

            if max_probability >= 0.6:
                final_name = self.encoder.inverse_transform(face_name)[0]
            else:
                final_name = "UNKNOWN"

            

            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 10, 150), 10)
            # cv2.putText(frame, str(final_name), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
            #             1, (0, 0, 255), 3, cv2.LINE_AA)
