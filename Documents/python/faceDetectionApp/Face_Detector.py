import cv2
from random import randrange

#open cv using cascade algorthim
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#image added
# img= cv2.imread('abj.png')
img= cv2.imread('multiple-faces.jpg')

grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


face_coordinates = trained_face_data.detectMultiScale(gray_img)
#print(face_coordinates)


#normally changed the static
# (x,y,w,h) = face_coordinates[0]
# cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

# (x,y,w,h) = face_coordinates[1]
# cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)   

#loop run
for(x,y,w,h) in face_coordinates:
  cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)


#showing Image
cv2.imshow('Showing Img',img)
# cv2.imshow('Gray', gray_img)

#wait for the above image
cv2.waitKey()
print("code completed")