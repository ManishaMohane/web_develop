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

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
print(APP_ROOT)
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
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

#@app.route('/welcome')
#@login_required
#def welcome():
 #return render_template('welcome.html')

#@app.route('/admin')
#@login_required
#def admin():
 #return render_template('admin.html')
@app.route('/admin')
def admin(): 
 #ID=input("enter your id")
 #name=str(input("enter your name"))
 #ID = str(request.form['id'])
 #name = str(request.form['name'])
 #update(ID,name)  
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

#@app.route('/admin', methods=['GET', 'POST'])
#@login_required
#def admin():
 #   if request.method == 'POST':
  #      session['username'] = request.form['username']
   #     return redirect(url_for('home'))
    #return '''
     #   <form action="" method="post">s
      #      <p>Username <input name="username">
       #     <p><button>Sign in</button></p>
       # </form>
    #'''

@app.route('/home')
@login_required
def home():
 return render_template('index3.html')

@app.route('/sef')
def sef():
 return render_template('search.html')

#@app.route('/sef1')
#def sef1():
 #return render_template('search1.html')

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

#@app.route('/rec1')
#def rec1(): 
 #g.db = connect_db() 
 #cur = g.db.execute('select id,name,class,emailid,phoneno,rollno from students') 
 #row = cur.fetchall()  
 #return render_template('index1.html',row=row)
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

#@app.route('/ser1',methods=['POST'])
#def ser1():
 
 #g.db = connect_db()
 #cur = g.db.execute( "select * from students where name = ? ", (request.form['search'],) )
 #row = cur.fetchall()
 #return render_template("index1.html",row=row)

@app.route('/logout')
@login_required
def logout():
 session.pop('logged_in',None)
 flash('!!You were just logged out')
 return redirect(url_for('login'))
@app.route('/updation', methods=['POST'])
def update(ID):
 
 g.db = connect_db()
 cur = g.db.cursor()
 cmd= "SELECT * FROM attendance WHERE ID="+str(ID)
 cur.execute(cmd)
 isRecordExist=0
 for row in cursor:
     isRecordExist=1
 if(isRecordExist==1):
 #g.db.execute("UPDATE attendance SET ATTENDANCE = 'PRESENT' WHERE NAME ="+str(name))
 #g.db.execute("UPDATE attendance SET ATTENDANCE = 'PRESENT', Timestamp = DATETIME('now') WHERE NAME=' "+name+" ' AND ID="+ID) 
    cmd1= "UPDATE attendance SET ATTENDANCE = 'PRESENT', Timestamp = DATETIME('now') WHERE ID="+ID
 else:
    cmd1="INSERT INTO attendance(ID) Values"+str(Id)
 cur.execute(cmd1)
 g.db.commit()
 cur1 = cur.execute( "select * from attendance")
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
 #cursor = conn.cursor()
 #c = conn.cursor()
 #update(ID,name) 
 #result = g.db.execute("SELECT * from students where ID="+ ID)
 #print (result)
 #if(result == 1):
  #  return render_template("signup.html",title="SignUp",uname=user,msg="already present")
 g.db.execute('INSERT INTO students (id,name,class,emailid,phoneno,rollno) VALUES(?,?,?,?,?,?)',(ID,name,class1,emailid,phoneno,rollno))
 update(ID)
 #g.db.execute("UPDATE attendance SET ATTENDANCE = 'PRESENT', Timestamp = DATETIME('now') WHERE ID="+ID) 

 #g.db.execute("UPDATE attendance SET ATTENDANCE = 'ABSENT', Timestamp = DATETIME('now') WHERE NAME=' "+name+" ' AND ID="+ID)  
 #g.db.execute("UPDATE attendance SET ATTENDANCE = 'PRESENT', Timestamp = DATETIME('now') WHERE NAME='manisha'") 
 
 #g.db.execute("INSERT INTO students (username,password,email) VALUES(%s, %s, %s)",(user,paswd,email))
 #g.db.execute('INSERT INTO students (id,name,class,emailid,phoneno,rollno) VALUES(?,?,?,?,?,?)',[request.form['id'],request.form['name'],request.form['class'],request.form['emailid'],request.form['phoneno'],request.form['rollno']]);
 #if students.ID == attendance.ID:
 #g.db.execute("UPDATE attendance SET ATTENDANCE = 'PRESENT', Timestamp = DATETIME('now') WHERE NAME= AND ID="+str(request.form['id'])
 #else:
  #   print("ID unknown")
 #g.db.execute("UPDATE attendance SET ATTENDANCE = 'PRESENT', Timestamp = DATETIME('now') WHERE NAME=' "+str(request.form['name'])+" ' AND ID="+str(request.form['id'])) 
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
       cv2.imwrite("/home/manisa/Desktop/data/index.jpg",frame)

       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

    # Display the resulting frame
       cv2.imshow('Video', frame)

# When everything is done, release the capture
   video_capture.release()
   cv2.destroyAllWindows()
   return redirect(url_for('home'))  

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    # Check if a valid image file was uploaded
    #ID = str(request.form['id'])
    #session['id'] = ID
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
    #return render_template("stud.html")


def detect_faces_in_image(file_stream):
  
    video_capture = cv2.VideoCapture(0)
    img = face_recognition.load_image_file(file_stream)
    
    #def test():
    #	test_run=cv2.imread('1.jpg',1)
    #	test_run=cv2.resize(test_run,(160,160))
    #	test_run=test_run.astype('float')/255.0
    #	test_run=np.expand_dims(test_run,axis=0)
    #	test_run=e.calculate(test_run)
    #	test_run=np.expand_dims(test_run,axis=0)
    #	test_run=model.predict(test_run)[0]
    # Get face encodings for any faces in the uploaded image
    unknown_face_encodings = face_recognition.face_encodings(img)
    with open('dataset_faces.dat', 'rb') as f:
	    all_face_encodings = pickle.load(f)
    face_names = list(all_face_encodings.keys())
    face_encodings1 = np.array(list(all_face_encodings.values()))
    print(face_encodings1)
    face_found = False
    is_yours = False
    is_obama = False
    is_biden = False
    #ret=True
    #test()
    #known_face_encodings = [
   # obama_face_encoding,
    #manisha_face_encoding
    #]
    #known_face_names = [
    #"Barack Obama",
    #"Manisha"
     #]

     #if len(unknown_face_encodings) > 0:
      #  face_found = True
       # match_results = face_recognition.compare_faces(face_encodings, unknown_face_encodings)
        #if match_results[0]:
         #   is_obama = True
        #if match_results[1]:
         #   is_biden = True
        #if match_results[2]:
         #   is_yours = True
    # Return the result as json
       # names_with_result = list(zip(face_names, match_results))
        
    #result = {
#
 #       "face_found_in_image": face_found,
  #      "is_picture_of_obama": is_obama,
        
   #     "face_found_in_image": face_found,
    #    "is_picture_of_yours": is_yours,
    

     #   "face_found_in_image": face_found,
      #  "is_picture_of_biden": is_biden
    
#}
 #   return jsonify(result)
    # return (names_with_result)
    while True:
    # Grab a single frame of video
       ret, frame = video_capture.read()
       #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       #faces = faceCascade.detectMultiScale(
        #  gray,
         # scaleFactor=1.1,
         # minNeighbors=5,
         # minSize=(30, 30)
         #)
       #frame=cv2.flip(frame,1)
       #detected,x,y,w,h=fd.detectFace(frame)

       #if(detected is not None):
        #   f=detected
         #  detected=cv2.resize(detected,(160,160))
          # detected=detected.astype('float')/255.0
           #detected=np.expand_dims(detected,axis=0)
           #feed=e.calculate(detected)
           #feed=np.expand_dims(feed,axis=0)
           #prediction=model.predict(feed)[0]

           #result=int(np.argmax(prediction))
           #for i in people:
            #   if(result==i):
             #     label=people[i]
              #  if(a[i]==0):
               #     data.update(label)
                #a[i]=1
                #abhi=i

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
           #if len(unknown_face_encodings) > 0:
            #  face_found = True
             # match_results = face_recognition.compare_faces(face_encodings, face_encoding)

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
    
@app.route("/upload1", methods=['GET','POST']) 
def upload():
	target = os.path.join(APP_ROOT,"train/")
	if not os.path.isdir(target):
		os.mkdir(target)
	classfolder = str(request.form['class_folder'])
	session['classfolder'] = classfolder
	target1 = os.path.join(target,str(request.form["class_folder"])+"/")
	session['target1']=target1
	print(target1)
	model = os.path.join(APP_ROOT,"model/")
	if not os.path.isdir(model):
		os.mkdir(model)
	classname = str(request.form['class_folder']+"/")
	model = os.path.join(model,classname)
	if not os.path.isdir(model):
		os.mkdir(model)
	session['model']=model
	session['classname'] = classname
	if not os.path.isdir(target1):
		os.mkdir(target1)
	id_folder = str(request.form["id_folder"])
	session['id_folder']= id_folder
	target2 = os.path.join(target1,id_folder+"/")
	if not os.path.isdir(target2):
		os.mkdir(target2)
	target3 = os.path.join(target2,id_folder+"/")
	if not os.path.isdir(target3):
		os.mkdir(target3)
	for file in request.files.getlist("file"):
		print(file)
		filename = file.filename
		destination = "/".join([target3,filename])
		print(destination)
		file.save(destination)
	return call_train()

def call_train():
	id_folder = str(session.get('id_folder'))
	model=str(session.get('model'))
	if not os.path.isdir(model + id_folder):
		os.mkdir(model + id_folder)
	model = model + id_folder + "/"
	model = model + "model"
	target1=str(session.get('target1'))
	print(id_folder)
	print (target1)
	target1 = target1 +id_folder 
	print(target1)
	print(model)
	return train(train_dir=target1,model_save_path=model)

def train(train_dir, model_save_path = "", n_neighbors = None, knn_algo = 'ball_tree', verbose=True):
    id_folder=str(session.get('id_folder'))
    X = []
    y = []
    z = 0
    for class_dir in listdir(train_dir):
        if not isdir(join(train_dir, class_dir)):
            continue
        for img_path in image_files_in_folder(join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            faces_bboxes = face_locations(image)
            if len(faces_bboxes) != 1:
                if verbose:
                    print("image {} not fit for training: {}".format(img_path, "didn't find a face" if len(faces_bboxes) < 1 else "found more than one face"))
                    os.remove(img_path)
                    z = z + 1
                continue
            X.append(face_recognition.face_encodings(image, known_face_locations=faces_bboxes)[0])
            y.append(class_dir)
    print(listdir(train_dir+"/"+id_folder))
    train_dir_f = listdir(train_dir+"/"+id_folder)
    for i in range(len(train_dir_f)):
    	if(train_dir_f[i].startswith('.')):
    		os.remove(train_dir+"/"+id_folder+"/"+train_dir_f[i])

    print(listdir(train_dir+"/"+id_folder))
    
    if(listdir(train_dir+"/"+id_folder)==[]):
    	return render_template("upload.html",msg1="training data empty, upload again")
    elif(z >= 1):
    	return render_template("upload.html",msg1="Data trained for "+id_folder+", But one of the image not fit for trainning")
    if n_neighbors is None:
        n_neighbors = int(round(sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically as:", n_neighbors)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    if model_save_path != "":
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return render_template("upload.html",msg1="Data trained for "+ id_folder)

#while ret:
 #   ret,frame=cap.read()
  #  frame=cv2.flip(frame,1)
   # detected,x,y,w,h=fd.detectFace(frame)

    #if(detected is not None):
     #   f=detected
      #  detected=cv2.resize(detected,(160,160))
       # detected=detected.astype('float')/255.0
        #detected=np.expand_dims(detected,axis=0)
        #feed=e.calculate(detected)
        #feed=np.expand_dims(feed,axis=0)
        #prediction=model.predict(feed)[0]

        #result=int(np.argmax(prediction))
        #for i in people:
         #   if(result==i):
          #      label=people[i]
           #     if(a[i]==0):
            #        data.update(label)
             #   a[i]=1
              #  abhi=i

        #data.update(label)
        #cv2.putText(frame,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        #if(a[abhi]==1):
         #   cv2.putText(frame,"your attendance is complete",(x,y-30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(252,160,39),3)
        #cv2.imshow('onlyFace',f)
    #cv2.imshow('frame',frame)
    #if(cv2.waitKey(1) & 0XFF==ord('q')):
     #   break
#cap.release()
#cv2.destroyAllWindows()
    
if __name__ == '__main__':
 app.run(debug=True)
