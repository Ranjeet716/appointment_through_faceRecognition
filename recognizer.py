import cv2
import os
import numpy as np

#This module contains all common functions that are called in tester.py file
def check():
    print("In recognition part")
#Given an image below function returns rectangle for face detected alongwith gray scale image
def faceDetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)#convert color image to grayscale
    face_haar_cascade=cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_default.xml')#Load haar classifier
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=5)#detectMultiScale returns rectangles
    #where the face is detected in particular test image
    #while we train our classifier it must train for specific images with certain size decleared
    #we do rescaling so that to increasinse the detection probability by classifier 
    #min neighbour means  when mean neigh is 0 than there are lot of false posistive and lot of false posistive images 
    #
    return faces,gray_img

#Given a directory below function returns part of gray_img which is face alongwith its label/ID
def image_level(directory):
    #face list will contail region of face or face area from gray scale image
    face_list=[]
    #sub_dir is the subdirectory where various images of one type is stored...no of sub directory is same as no of level..and images of one type is laveled to a particulat level
    sub_dir=[]
    #in first level it explore forst root....in second loop it explore all the roots present in the second level just like it do with root level
    #here path is the dir name sub dir is sub directory present inside dir name and filename is the files present inside the dir name so this is for the first loop...in second loop next level is explored and in next level third level is explored
    for path,subdirc,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system file")#Skipping files that startwith.
                continue
            id=os.path.basename(path)#fetching subdirectory names
            img_path=os.path.join(path,filename)#fetching image path
            print("img_path:",img_path)
            print("id:",id)
            test_img=cv2.imread(img_path)#loading each image one by one
            if test_img is None:
                print("Image not loaded properly")
                continue
            faces_rect,gray_img=faceDetection(test_img)#Calling faceDetection function to return faces detected in particular image
            if len(faces_rect)!=1:
               continue #Since we are assuming only single person images are being fed to classifier
            (x,y,w,h)=faces_rect[0]
            roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from grayscale image
            face_list.append(roi_gray)
            sub_dir.append(int(id))
    return face_list,sub_dir


#Below function trains haar classifier and takes faces,faceID returned by previous function as its arguments
def train_classifier(face_list,sub_dir):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(face_list ,np.array(sub_dir))
    return face_recognizer

#Below function draws bounding boxes around detected face in image
def draw_rect(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,225,0),thickness=1)

#Below function writes name of person for detected label
def put_text(test_img,text,x,y):
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,2,(0,225,0),1)
