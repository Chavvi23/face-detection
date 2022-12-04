import streamlit as st
import cv2
from PIL import Image
import numpy as np

try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
except Exception:
    st.write("Error loading cascade classifiers")

def detect(image):
    
    image = np.array(image.convert('RGB'))
    faces = face_cascade.detectMultiScale(image=image, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        
       
        cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        
        roi = image[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi)
        
        smile = smile_cascade.detectMultiScale(roi, minNeighbors = 25)
        
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
            
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi, (sx, sy), (sx+sw, sy+sh), (0,0,255), 2)

    return image, faces


def main():
    st.title("WELCOME TO FACE DETECTION APP :camera_with_flash: ")
    st.write("**This web application is built using the Haar cascade Classifier**")
    st.write("Upload the image in the specified format and click detect to process and identify faces in the image")
    image_file = st.file_uploader("UPLOAD IMAGE", type=['jpeg', 'png', 'jpg', 'webp'])

    if image_file is not None:

    	image = Image.open(image_file)

    	if st.button("Detect"):
    		result_img, result_faces = detect(image=image)
    		st.image(result_img, use_column_width = True)
    		st.success("Found {} face(s)!\n".format(len(result_faces)))
if __name__ == "__main__":
    main()
