import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import face_recognition

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0
cam = cv2.VideoCapture(0)
harcascadePath = "haarcascade_frontalface_default.xml"
detector=cv2.CascadeClassifier(harcascadePath)
def insertOrUpdate(Id,Name):
     conn=sqlite3.connect("test1.db")
     cmd= "SELECT * FROM attendance WHERE ID="+str(Id)
     cursor=conn.execute(cmd)
     isRecordExist=0
     for row in cursor:
       isRecordExist=1
     if(isRecordExist==1):
       cmd="UPDATE attendance SET Name="+str(Name)+"WHERE ID="+str(Id)
     else:
       cmd="INSERT INTO attendance(Id,Name) Values("+str(Id)+","+str(Name)+")"
     conn.execute(cmd)
     conn.commit()
     conn.close()
id=input("enter your id")
name=input("enter your name")
insertOrUpdate(id,name)  
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    #font = cv2.FONT_HERSHEY_SIMPLEX(
     #       thickness=1,
      #      white=(255,) * 3,
       #     size=1
        #    left = (x, y+h+12)
     #)
                 
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.putText(frame, 'IDENTITY UNKNOWN', left, font, size, white, thickness)
        #cv2.putText(frame, 'FACE ID', (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,) * 3, 1)
        face_no = faces.shape[0];
    for face_no, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.putText(frame, str(face_no), (x+150, y+h+30), cv2.FONT_HERSHEY_TRIPLEX, .7, (0, 0, 0), 1, cv2.LINE_AA)
    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)
    cv2.imwrite("/home/manisa/Desktop/data/index.jpg",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
