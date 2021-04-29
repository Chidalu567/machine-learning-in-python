import cv2
import face_recognition
import numpy as np
import os

path=r'C:\Users\chidalu craving\Desktop\Face Recognition in python\Attendance Project\Original Image';

har_face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml'); #do a cascade classifier object

images=[]; #python list definition
image_names=[]; #Python list definition
image_list=os.listdir(path); #give the list directory of path
print(image_list)

for obj in image_list:
    img=cv2.imread(r'{}\{}'.format(path,obj),cv2.IMREAD_COLOR); #Image read
    images.append(img); #append value to list
    image_names.append(os.path.splitext(obj)[0]);#append value tolist

print(image_names)


#Second setp is to encode the image
def encode_image(images): #function definition
    encoded_image_list=[]; #python list definition
    for img in images:
        curimage=cv2.cvtColor(img,cv2.COLOR_BGR2RGB); #convert color of image from bgr2rgb

        encode=face_recognition.face_encodings(curimage)[0]; #do a face_encodings
        encoded_image_list.append(encode); #append value tolist
    return encoded_image_list;

encodedlistknown=encode_image(images); #function call

'''We do a test for the match'''
cap=cv2.VideoCapture(0); #we create a capture object

while cap.isOpened():
    ret,frame=cap.read(); #read frame by frame


    imgS=cv2.resize(frame,(0,0),None,0.25,0.25); #resize frame
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB); #convert color of image from bgr2rgb

    face_loc=face_recognition.face_locations(imgS);#find face location
    face_encode=face_recognition.face_encodings(imgS,face_loc); #face encodings on image with a face_location

    for face,encode in zip(face_loc,face_encode):
        match=face_recognition.compare_faces(encodedlistknown,encode); #compare faces
        face_dis=face_recognition.face_distance(encodedlistknown,encode); #find face distance

        matchindex=np.argmin(face_dis); #get argument minimum of list

        if match[matchindex]:
            name=image_names[matchindex].upper(); #get the value in uppercase

            y1,x2,y2,x1=face;
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4;
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2); #draw rectangle on frame
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED); #draw a rectangle on frame
            cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2); #putText on frame

    cv2.namedWindow('WebCam',cv2.WINDOW_NORMAL); #create a named window
    cv2.imshow('WebCam',frame); #image show

    if cv2.waitKey(10)&0xFF==ord('q'):
        break;

cv2.destroyWindow('WebCam'); #destroywindow
