import cv2
import os
import numpy as np
import recognizer as r

#just to check Whether our recognizer work or not so we import our recognizer module inside our test program  as r and to check
#module is imported successfully or not.
r.check()

test_img=cv2.imread('TestImages/frame0.jpg')
faces_detected,gray_img=r.faceDetection(test_img)
print("faces_detected is:",faces_detected)
#print("And its Gray form is ",gray_img)

#we can comment the following code,if we want to use the whole code futher it further...as it is saved as training.yml
faces,sub_dir=r.image_level('trainingImages')
face_recognizer=r.train_classifier(faces,sub_dir)
face_recognizer.write('trainingData.yml')


#Uncomment below line for subsequent runs
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')#use this to load training data for subsequent runs

name={1:"ranjeet"}
#x and y are starting coordinate and L=x+w is how long it will be(Lenght alog x-axis) and H= y+h is how high the image will be

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print("confidence:",confidence)
    print("label:",label)
    r.draw_rect(test_img,face)
    predicted_name=name[label]
    if(confidence>60):
        continue
    r.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(1000,1000))
cv2.imshow("face dtecetion tutorial",resized_img)
cv2.waitKey(0)#Waits indefinitely until a key is pressed
cv2.destroyAllWindows
