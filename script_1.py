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
import datetime as dt
from time import sleep
import pandas as pd
import csv
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
 ID=input("enter your id")
 name=str(input("enter your name"))
 update(ID,name)  
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
 cur = g.db.execute( "select * from students where name = ? ", (request.form['search'],) )
 row = cur.fetchall()
 return render_template("index.html",row=row)


@app.route('/ser1',methods=['POST'])
def ser1():
 
 g.db = connect_db()
 cur = g.db.execute( "select * from students where name = ? ", (request.form['search'],) )
 row = cur.fetchall()
 return render_template("index1.html",row=row)

@app.route('/logout')
@login_required
def logout():
 session.pop('logged_in',None)
 flash('!!You were just logged out')
 return redirect(url_for('login'))

def update(ID,name):
 
 g.db = connect_db()
 cmd= "SELECT * FROM attendance WHERE ID="+str(ID)
 g.db.execute(cmd)
 #g.db.execute("UPDATE attendance SET ATTENDANCE = 'PRESENT' WHERE NAME ="+str(name))
 g.db.execute("UPDATE attendance SET ATTENDANCE = 'PRESENT', Timestamp = DATETIME('now') WHERE NAME=' "+str(name)+" ' AND ID="+str(ID)) 
 g.db.commit()
 cur = g.db.execute( "select * from attendance")
 row = cur.fetchall()
 return render_template("file1.html",row=row)

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
 return redirect(url_for('home'))

@app.route('/search')
def search():
 return render_template("search.html")
def connect_db():
 return sqlite3.connect(app.database)

@app.route('/Take_Images') 
def Take_Images():
   cam = cv2.VideoCapture(0)
   harcascadePath = "haarcascade_frontalface_default.xml"
   detector=cv2.CascadeClassifier(harcascadePath)
   def insertOrUpdate(id,name):
      conn=sqlite3.connect("test1.db")
      cmd= "SELECT * FROM attendance WHERE id="+str(id)
      cursor=conn.execute(cmd)
      isRecordExist=0
      for row in cursor:
        isRecordExist=1
      if(isRecordExist==1):
        cmd="UPDATE attendance SET NAME="+str(name)+"WHERE id="+str(id)
      else:
        cmd="INSERT INTO attendance(id,Name) Values("+str(id)+","+str(name)+")"
      conn.execute(cmd)
      conn.commit()
      conn.close()
   id=input("enter your id")
   name=input("enter your name")
   insertOrUpdate(id,name)  
   sampleNum=0
   while(True):
      ret, img = cam.read()
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = detector.detectMultiScale(gray, 1.3, 5)
      for (x,y,w,h) in faces:     
       sampleNum=sampleNum+1
       cv2.imwrite("data/ "+name +"."+id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
       cv2.rectangle(img,(x-50,y-50),(x+w+50,y+h+50),(255,0,0),2)
      cv2.imshow('frame',img)
            #wait for 100 miliseconds 
      if cv2.waitKey(100) & 0xFF == ord('q'):
           break
            # break if the sample number is morethan 100
      elif sampleNum>60:
           break
      cam.release()
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
  
    # Load the uploaded image file
    img = face_recognition.load_image_file(file_stream)
    # Get face encodings for any faces in the uploaded image
    unknown_face_encodings = face_recognition.face_encodings(img)
    with open('dataset_faces.dat', 'rb') as f:
	    all_face_encodings = pickle.load(f)
    face_names = list(all_face_encodings.keys())
    face_encodings = np.array(list(all_face_encodings.values()))
    print(face_encodings)
    face_found = False
    is_yours = False
    is_obama = False
    is_biden = False
    if len(unknown_face_encodings) > 0:
        face_found = True
        match_results = face_recognition.compare_faces(face_encodings, unknown_face_encodings)
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
    return (names_with_result)
if __name__ == '__main__':
 app.run(debug=True)
