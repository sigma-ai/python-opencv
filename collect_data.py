import cv2
import os

cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#get frame width and height for video
width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)


# For each person, enter one numeric face id
name = input('\n Enter your name and press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

#define the codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
out = cv2.VideoWriter(name+'.mp4', fourcc, 20.0, (width, height))

count = 0
while(True):
    ret, img = cam.read()
    #img = cv2.flip(img, -1) # flip video image vertically
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(img, 1.6, 5)
    out.write(img)
    cv2.imshow('img', img)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(name) + '.' + str(count) + ".jpg", img[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 100: # Take 30 face sample and stop video
         break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
