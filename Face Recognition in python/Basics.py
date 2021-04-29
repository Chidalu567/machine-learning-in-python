import cv2
import face_recognition

'''
1]Load image
2]convert color of image from bgr2rgb
3]find the location
4]find the encodings
5]compare the images
6]Find the distance

'''

img=face_recognition.load_image_file('images/chidalu.jpg'); #load imagefrom file
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB); #convert color of image from bgr2rgb

imgtest=face_recognition.load_image_file('images/chidalu2.jpg'); #load image file
imgtest=cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB); #convert color of image from bgr2rgb

face_loc=face_recognition.face_locations(img)[0]; #find the face location of image
encoding_img=face_recognition.face_encodings(img)[0]; #find face encodings of image
cv2.rectangle(img,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(255,0,0),3); #draw rectangle on image

test_loc=face_recognition.face_locations(imgtest)[0]; #face_locations of image
test_encodings=face_recognition.face_encodings(imgtest)[0]; #face_encodings of image test
cv2.rectangle(imgtest,(test_loc[3],test_loc[0]),(test_loc[1],test_loc[2]),(255,0,0),3); #draw rectangle on image

results=face_recognition.compare_faces([encoding_img],test_encodings); #compare faces
print(results)

dis=face_recognition.face_distance([encoding_img],test_encodings); #face_distance

cv2.putText(imgtest,"{} .\n {}".format(results,round(dis[0],2)),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255)); #put text on image


cv2.imshow('Original',img); #Image show
cv2.imshow('Test',imgtest); #Image show

cv2.waitKey(0)

