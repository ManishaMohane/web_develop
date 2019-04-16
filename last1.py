import face_recognition
from flask import Flask,render_template,g,request,flash,redirect,url_for,session,flash,jsonify,abort
from werkzeug import secure_filename
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from functools import wraps
from sklearn import neighbors
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from os import listdir 
from os.path import isdir, join, isfile, splitext
import sqlite3
import os
import numpy as np
import pickle
import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import pandas as pd
import csv
from face_recognition import face_locations
from face_recognition.face_recognition_cli import image_files_in_folder
from flask_mail import Mail,  Message
#import lable_image

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
print(APP_ROOT)
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__)

app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# mail settings

app.config.update(
    #DEBUG = True,
    #Email settings
    MAIL_SERVER = 'smtp.gmail.com',
    MAIL_PORT = 465,
    MAIL_USE_SSL = True,
    MAIL_USERNAME = 'manishamohane54@gmail.com',
    MAIL_PASSWORD = 'xscu iqbj Wecn ailn',
   #MAIL_DEFAULT_SENDER = 'manishamohane54@gmail.com'
    )
mail = Mail(app)

def login_required(f):
   @wraps(f)
   def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('You need to login first.')
            return redirect(url_for('login'))
   return wrap

app.secret_key = os.urandom(24)
app.database='test1.db'
conn=sqlite3.connect('test1.db')
@app.route('/admin')
def admin(): 
 g.db = connect_db() 
 cur = g.db.execute('select id,name,attendance,Timestamp from attendance')        
 row = cur.fetchall()  
 return render_template('file1.html',row=row)

@app.route('/add1', methods=['POST'])
@login_required
def add1():
 g.db=connect_db()
 g.db.execute('INSERT INTO attendance (id,name,attendance,Timestamp) VALUES(?,?,?,?)',[request.form['id'],request.form['name'],request.form['attendance'],request.form['Timestamp']]);
 g.db.commit()
 flash('posted')
 #return redirect(url_for('home'))
 return redirect(url_for('home'))
@app.route('/home')
@login_required
def home():
 return render_template('index3.html')

@app.route('/sef')
def sef():
 return render_template('search.html')

@app.route('/sef2')
def sef2():
 return render_template('search2.html')

@app.route('/del')
@login_required
def delt():
 return render_template('del.html')

@app.route('/stud', methods=['GET','POST'])
@login_required
def stud():
 return render_template('stud.html')

@app.route('/', methods=['GET','POST'])
def login():
   error = None
   if request.method == 'POST':
     if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
     else:
         session['logged_in']=True 
         return redirect(url_for('home'))
   flash('!You were just logged in!!')
   return render_template('front1.html', error=error)

@app.route('/rec')
def rec(): 
 g.db = connect_db() 
 cur = g.db.execute('select id,name,class,emailid,phoneno,rollno from students')
 
 row = cur.fetchall()  
 return render_template('index.html',row=row)

@app.route('/rec2')
def rec2(): 
 g.db = connect_db() 
 cur = g.db.execute('select id,name,attendance,Timestamp from attendance')
 print(cur)
 row = cur.fetchall()  
 return render_template('file1.html',row=row)

@app.route('/ser',methods=['POST'])
def ser():
 
 g.db = connect_db()
 cur = g.db.execute( "select * from students where name = ? ", (request.form['search'],) )
 row = cur.fetchall()
 return render_template("index.html",row=row)

@app.route('/ser2',methods=['POST'])
def ser2():
 
 g.db = connect_db()
 cur = g.db.execute( "select * from attendance where name = ? ", (request.form['search'],) )
 row = cur.fetchall()
 return render_template("file1.html",row=row)

@app.route('/logout')
@login_required
def logout():
 session.pop('logged_in',None)
 flash('!!You were just logged out')
 return redirect(url_for('login'))

#@app.route('/updation', methods=['POST'])
def update(ID):
 g.db = connect_db()
 #cur = g.db.cursor()
 cmd= "SELECT * FROM attendance WHERE ID="+str(ID)
 g.db.execute(cmd)
 isRecordExist=0
 #for row in cursor:
  #   isRecordExist=1
 #if(isRecordExist==1):
 cmd1= "UPDATE attendance SET ATTENDANCE = 'PRESENT', Timestamp = DATETIME('now') WHERE ID="+ID
 #else:
  #  cmd1="INSERT INTO attendance(ID) Values"+str(Id)
 g.db.execute(cmd1)
 g.db.commit()
 cur1 = g.db.execute( "select * from attendance")
 row = cur1.fetchall()
 return render_template("stud.html",row=row)

@app.route('/delete',methods=['POST'])
@login_required
def delete():
 g.db = connect_db()
 g.db.execute( "delete from students where name = ? ", (request.form['delete'],) )
 g.db.commit()
 cur=g.db.execute( "select * from students ")
 row=cur.fetchall()
 return render_template("delete.html",row=row) 

@app.route('/add', methods=['POST'])
@login_required
def add():
 g.db=connect_db()
 ID = str(request.form['id'])
 name = str(request.form['name'])
 class1 = str(request.form['class'])
 emailid = str(request.form['emailid'])
 phoneno = str(request.form['phoneno'])
 rollno = str(request.form['rollno'])
 g.db.execute('INSERT INTO students (id,name,class,emailid,phoneno,rollno) VALUES(?,?,?,?,?,?)',(ID,name,class1,emailid,phoneno,rollno))
 update(ID)
 g.db.commit()
 flash('posted')
 # return redirect(url_for('home'))
 return redirect(url_for('home'))

@app.route('/search')
def search():
 return render_template("search.html")
def connect_db():
 return sqlite3.connect(app.database)

@app.route('/search2')
def search2():
 return render_template("search2.html")
def connect_db():
 return sqlite3.connect(app.database)

def send_mail(ID):
    now = dt.datetime.now()
    print(now)
    msg = mail.send_message(
        'Send Mail tutorial!',
        sender='manishamohane54@gmail.com',
        recipients=['manishamohane54@gmail.com'],
        body="student having ID "+str(ID)+" was present on "+ now.strftime("%Y-%m-%d %H:%M:%S")
    )
    return 'Mail sent'

@app.route('/Track_Image')
def TrackImages():
   video_capture = cv2.VideoCapture(0)
   with open('dataset_faces.dat', 'rb') as f:
	    all_face_encodings = pickle.load(f)
   face_names = list(all_face_encodings.keys())
   face_encodings1 = np.array(list(all_face_encodings.values()))

   while True:
    # Grab a single frame of video
      ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
      rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
      face_locations = face_recognition.face_locations(rgb_frame)
      face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
      for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
         matches = face_recognition.compare_faces(face_encodings1, face_encoding)

         name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
         if True in matches:
             first_match_index = matches.index(True)
             name = face_names[first_match_index]

        # Draw a box around the face
         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
         font = cv2.FONT_HERSHEY_DUPLEX
         cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
      cv2.imshow('Video', frame)
      cv2.imwrite("/home/manisa/Desktop/data/image1.jpg",frame)
    # Hit 'q' on the keyboard to quit!
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

# Release handle to the webcam
   video_capture.release()
   cv2.destroyAllWindows()
# Load a sample picture and learn how to recognize it.

   return redirect(url_for('home'))  

@app.route('/Take_Images')
def Take_Images():
   cascPath = "haarcascade_frontalface_default.xml"
   faceCascade = cv2.CascadeClassifier(cascPath)
   log.basicConfig(filename='webcam.log',level=log.INFO)

   video_capture = cv2.VideoCapture(0)
   anterior = 0

   while True:
       if not video_capture.isOpened():
          print('Unable to load camera.')
          sleep(5)
          pass

    # Capture frame-by-frame
       ret, frame = video_capture.read()
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       image_recog = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       faces = faceCascade.detectMultiScale(
          gray,
          scaleFactor=1.1,
          minNeighbors=5,
          minSize=(30, 30)
         )

    # Draw a rectangle around the faces
       for (x, y, w, h) in faces:
         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.putText(frame, 'IDENTITY UNKNOWN', left, font, size, white, thickness)
         cv2.putText(frame, 'FACE ID', (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,) * 3, 1)
         face_no = faces.shape[0];
       for face_no, (x, y, w, h) in enumerate(faces):
         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
         cv2.putText(frame, str(face_no), (x+150, y+h+30), cv2.FONT_HERSHEY_TRIPLEX, .7, (0, 0, 0), 1, cv2.LINE_AA)
       if anterior != len(faces):
         anterior = len(faces)
         log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

    # Display the resulting frame
       cv2.imshow('Video', frame)
       cv2.imwrite("/home/manisa/Desktop/data/index.jpg",image_recog)

       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

    # Display the resulting frame
       #cv2.imshow('Video', frame)

# When everything is done, release the capture
   video_capture.release()
   cv2.destroyAllWindows()
   return redirect(url_for('home'))  

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) # The image file seems valid! Detect faces and return the result.
            
            return detect_faces_in_image(file)


    # If no valid image file was uploaded, show the file upload form:
    return """
    <!doctype html>
    <title>Is this a picture of yours?</title>
    <h1>Upload a picture</h1>
    <form method="POST" enctype="multipart/form-data">
    <input type="file" name="file">
    <input type="submit" value="Upload">
    </form>
    
    """    
def detect_faces_in_image(file_stream):
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    video_capture = cv2.VideoCapture(0)
    img = face_recognition.load_image_file(file_stream)
   # X_img = face_recognition.load_image_file(X_img_path)
   # X_faces_loc = face_locations(img)
    # Get face encodings for any faces in the uploaded image
    unknown_face_encodings = face_recognition.face_encodings(img)
    with open('dataset1.dat', 'rb') as f:
	    all_face_encodings = pickle.load(f)
    face_names = list(all_face_encodings.keys())
    face_encodings1 = np.array(list(all_face_encodings.values()))
    print(face_encodings1)
    face_found = False
    is_yours = False
    is_obama = False
    is_biden = False
    
    while True:
    # Grab a single frame of video
       ret, frame = video_capture.read()
       rgb_frame = frame[:, :, ::-1]

      # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       faces = faceCascade.detectMultiScale(
       #   gray,
	  rgb_frame,
          scaleFactor=1.1,
          minNeighbors=5,
          minSize=(30, 30)
         )
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
       #rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
       #face_locations = face_recognition.face_locations(rgb_frame)
       #face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
       #for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
       for (x, y, w, h) in faces:
        # See if the face is a match for the known face(s) 
           matches = face_recognition.compare_faces(face_encodings1, unknown_face_encodings)

           name = "Unknown"
           #if len(unknown_face_encodings) > 0:
            #  face_found = True
             # match_results = face_recognition.compare_faces(face_encodings, face_encoding)

        # If a match was found in known_face_encodings, just use the first one.
           if True in matches:
                first_match_index = matches.index(True)
                name = face_names[first_match_index]
 
        # Draw a box around the face
           #cv2.rectangle(frame, (h, x), (y, w), (0, 0, 255), 2)
           cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Draw a label with a name below the face
           cv2.rectangle(frame, (h, w - 35), (y, w), (0, 0, 255), cv2.FILLED)
           font = cv2.FONT_HERSHEY_DUPLEX
           cv2.putText(frame, name, (h + 6, w - 6), font, 1.0, (255, 255, 255), 1)
           ID = name
           update(ID)
           send_mail(ID)
           #label_image = model.tf("/home/manisa/desktop/web_develop/tf_files")
    #       now = datetime.datetime.now()
     #      msg = mail.send_message(
      #                       'Send Mail tutorial!',
       #                       sender='manishamohane54@gmail.com',
        #                      recipients=['manishamohane54@gmail.com'],
         #                     body="student having ID"+str(ID)+"was present on"+ now.strftime("%Y-%m-%d %H:%M:%S")
          #                      )
       print("message sent")
    # Display the resulting image
       cv2.imshow('Video', frame)
       cv2.imwrite("/home/manisa/Desktop/data/image1.jpg",frame)
    # Hit 'q' on the keyboard to quit!
       if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
# Load a sample picture and learn how to recognize it.

    #return redirect(url_for('send_mail')) 
    return redirect(url_for('home'))

# @app.route('/send_mail')


# route to send emails to parents and students

#@app.route('/send_mail',methods=['POST'])
#def send_mail():
    #test_append = str(request.form['folder_name'])
    #teacher_name = str(session.get('user'))
    #excel_dir = APP_ROOT+"/excel/"+test_append+"/"+teacher_name+"/"
    #excel_date = request.form['fname']
    #time = request.form['ftime']
    #time = time[:2]
    #final_send = glob(excel_dir + "/" + excel_date+ "@" + time +"*.xlsx")[0]
    #print(final_send)
    #df = pd.read_excel(final_send)
    #roll_id = list(df['Roll Id'])
 #   ID = 
  #  print(type(roll_id))
   # print(roll_id)
    #cursor = conn.cursor()
   # for i in range(len(roll_id)):
    #    cursor.execute("SELECT student_email,parent_email from student_login where binary roll_id=%s",[roll_id[i]])
     #   email = list(cursor.fetchone())
      #  print(type(email[1]))
       # print(email[0])
        #print(email[1])
        #msg = Message('Auto Generated',recipients= [email[0],email[1]])
        #msg.body = "Hi.. " + roll_id[i] + " is present for the lecture of " + "Prof. " +str(teacher_name.split('.',1)[0]) + ", which is held on " + excel_date + "@" + time + "hrs"
        #msg.html = "Hi.. " + roll_id[i] + " is present for the lecture of " + "Prof. " +str(teacher_name .split('.',1)[0])+ ", which is held on " + excel_date + "@" + time + "hrs"
        #mail.send(msg)
    #return "<h1>mail sent<h1>"
    
if __name__ == '__main__':
 app.run(debug=True)
