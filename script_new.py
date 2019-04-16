import face_recognition
from flask import Flask,render_template,g,request,flash,redirect,url_for,session,flash,jsonify,abort
from werkzeug import secure_filename
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from functools import wraps
import sqlite3
import os
import numpy as np
import pickle
import cv2
import sys
import logging as log
import datetime 
from time import sleep
import pandas as pd

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
 g.db = connect_db() 
 cur = g.db.execute('select id,name,attendance,Timestamp from attendance')        
 row = cur.fetchall()  
 return render_template('file1.html',row=row)
@app.route('/rec2')
def rec2(): 
 g.db = connect_db() 
 cur = g.db.execute('select id,name,attendance,Timestamp from attendance')
 
 row = cur.fetchall()  
 return render_template('file1.html',row=row)
@app.route('/add1', methods=['POST'])
@login_required
def add1():
 g.db=connect_db()
 g.db.execute('INSERT INTO attendance (id,name,attendance,datetime) VALUES(?,?,?,?)',[request.form['id'],request.form['name'],request.form['attendance'],request.form['datetime']]);
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
     #   <form action="" method="post">
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

@app.route('/sef1')
def sef1():
 return render_template('search1.html')
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
@app.route('/rec1')
def rec1(): 
 g.db = connect_db() 
 cur = g.db.execute('select id,name,class,emailid,phoneno,rollno from students')
 
 row = cur.fetchall()  
 return render_template('index1.html',row=row)
@app.route('/ser',methods=['POST'])
def ser():
 
 g.db = connect_db()
 cur=g.db.execute( "select * from students where name = ? ", (request.form['search'],) )
 row = cur.fetchall()
 return render_template("index.html",row=row)


@app.route('/ser1',methods=['POST'])
def ser1():
 
 g.db = connect_db()
 cur=g.db.execute( "select * from students where name = ? ", (request.form['search'],) )
 row = cur.fetchall()
 return render_template("index1.html",row=row)

@app.route('/logout')
@login_required
def logout():
 session.pop('logged_in',None)
 flash('!!You were just logged out')
 return redirect(url_for('login'))

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
             
 g.db.execute('INSERT INTO students (id,name,class,emailid,phoneno,rollno) VALUES(?,?,?,?,?,?)',[request.form['id'],request.form['name'],request.form['class'],request.form['emailid'],request.form['phoneno'],request.form['rollno']]);
 
 g.db.commit()
 flash('posted')
 #return redirect(url_for('home'))
 return redirect(url_for('home'))

@app.route('/search')
def search():
 return render_template("search.html")
def connect_db():
 return sqlite3.connect(app.database)
#filedata = []
#with open(url_for('static', filename='bmidata.txt')) as f:
 #   for line in f:

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
    #lmm_image = face_recognition.load_image_file("MANISHA.jpg")
    #lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]

   # al_image = face_recognition.load_image_file("alex-lacamoire.png")
    #al_face_encoding = face_recognition.face_encodings(al_image)[0]

  #  known_faces = [
   # lmm_face_encoding
   # al_face_encoding
    #                 ]
  
    # Load the uploaded image file
    img = face_recognition.load_image_file(file_stream)
    # Get face encodings for any faces in the uploaded image
    unknown_face_encodings = face_recognition.face_encodings(img)
    #np.save('lmm_face_encoding.npy', lmm_face_encoding)
    #biden_face_encoding_loaded = np.load('lmm_face_encoding.npy')
    with open('dataset_faces.dat', 'rb') as f:
	    all_face_encodings = pickle.load(f)
    face_names = list(all_face_encodings.keys())
    face_encodings = np.array(list(all_face_encodings.values()))
    print(face_encodings)
    #face_encodings = np.array(all_face_encodings)
    face_found = False
    is_yours = False
    is_obama = False
    is_biden = False
    if len(unknown_face_encodings) > 0:
        face_found = True
        # See if the first face in the uploaded image matches the known face of Obama
        #l = ["face_encodings[0]", "face_encodings[1]", "face_encodings[2]"] 
        #for i in l: 
        match_results = face_recognition.compare_faces(face_encodings, unknown_face_encodings)
        #print(match_results)
        #match_results1 = face_recognition.compare_faces(face_encodings[1], unknown_face_encodings[0])
        #match_results2 = face_recognition.compare_faces(face_encodings[2], unknown_face_encodings[0])
        if match_results[0]:
            is_obama = True
        if match_results[1]:
            is_biden = True
        if match_results[2]:
            is_yours = True
    # Return the result as json
        names_with_result = list(zip(face_names, match_results))
        
    result = {
        "face_found_in_image": face_found,
        "is_picture_of_obama": is_obama,
        
        "face_found_in_image": face_found,
        "is_picture_of_yours": is_yours,
    

        "face_found_in_image": face_found,
        "is_picture_of_biden": is_biden
    
        }
    return jsonify(result)
    #return redirect(url_for('logout'))
    return (names_with_result)
if __name__ == '__main__':
 app.run(debug=True)
