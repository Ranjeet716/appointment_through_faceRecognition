import cv2
import os
import numpy as np
import recognizer as r

r.check()

test_img=cv2.imread('TestImages/frame0.jpg')
faces_detected,gray_img=r.faceDetection(test_img)
print("faces_detected is:",faces_detected)
print("And its Gray form is ",gray_img)


#Comment belows lines when running this program second time.Since it saves training.yml file in directory
faces,sub_dir=r.image_level('trainingImages')
face_recognizer=r.train_classifier(faces,sub_dir)
face_recognizer.write('trainingData.yml')


#Uncomment below line for subsequent runs
#face_recognizer=cv2.face.LBPHFaceRecognizer_create()
#face_recognizer.read('trainingData.yml')#use this to load training data for subsequent runs

name={0:"Priyanka",1:"Kangana",2:"ranjeet"}#creating dictionary containing names for each label
#x and y are starting cordinate and L=x+w is how long it will be and H= y+h is how high the image will be

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
    print("confidence:",confidence)
    print("label:",label)
    r.draw_rect(test_img,face)
    predicted_name=name[label]
    if(confidence>60):#If confidence more than 37 then don't print predicted face text on screen
        continue
    r.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(1000,1000))
cv2.imshow("face dtecetion tutorial",resized_img)
cv2.waitKey(0)#Waits indefinitely until a key is pressed
cv2.destroyAllWindows
