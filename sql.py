import sqlite3
conn = sqlite3.connect('test1.db')
print ("Opened database successfully");
c = conn.cursor()
#def update(self, name,ID):
 #          self.field_to_update = field_to_update
  #         self.value_to_set = value_to_set
   #        self.lec_no = lec_no
    #       self.myCursor.execute("UPDATE Lecturers SET "+self.field_to_update+" = (?) WHERE lec_no = (?)",(self.value_to_set,self.lec_no))       

     #      self.myConnection.commit() 
ID = input('Enter user id: ')
#cmd = "SELECT ID FROM students"
#c.execute(cmd)

#print(cmd)
#cmd= "SELECT * FROM attendance WHERE ID="+str(ID)
#var= "SELECT ID FROM students WHERE ID=1"
#cmd= "SELECT ID FROM attendance WHERE ID=1"
#c.execute(var)
#print(c1)
#c2=c.execute(cmd)
#print(c2)
#if(c1==c2):
 # print(cmd)
name = input('Enter name: ')
#cmd1 = "SELECT NAME FROM students"
#name=c.execute(cmd1)
#print(cmd1) 
#c.execute("SELECT students.name,attendance.name FROM students INNER JOIN attendance ON students.ID = attendance.ID")
#ALTER TABLE attendance RENAME TO attendance_new;
#c.execute('''CREATE TABLE attendance 
 #      (ID INTEGER PRIMARY KEY NOT NULL,
  #     NAME VARCHAR  NOT NULL,
   #    ATTENDANCE  VARCHAR  NOT NULL,
    #   Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
     #         );''')
#c.execute("ALTER TABLE 'attendance' ADD COLUMN COLNew 'date'")
#c.execute("DROP TABLE attendance")
#c.execute("INSERT INTO attendance VALUES ('1','shailja','absent','DATETIME('now')'")
#c.execute("SELECT * FROM attendance WHERE ID="+str(ID))
#c.execute("UPDATE attendance SET Name=' "+str(name)+" ' WHERE ID="+str(ID))
c.execute("UPDATE attendance SET ATTENDANCE = 'ABSENT', Timestamp = DATETIME('now') WHERE NAME=' "+str(name)+" ' AND ID="+str(ID))
#print(c.fetchone());
#c.execute("UPDATE attendance SET ATTENDANCE = 'ABSENT', Timestamp = DATETIME('now') WHERE NAME=' "+str(name)+" ' AND ID="+str(ID)) 
#c.execute("UPDATE attendance SET ATTENDANCE = 'PRESENT', Timestamp = DATETIME('now') WHERE NAME IN {'shailja','monika','manisha'} ")
#c.execute("UPDATE attendance SET ATTENDANCE = 'ABSENT' WHERE NAME ="+str('name'))
#c.execute("INSERT INTO attendance VALUES ('1','shailja','ABSENT','DEFAULT')")
#c.execute("INSERT INTO attendance VALUES ('2','monika','ABSENT','DEFAULT')")
#c.execute("INSERT INTO attendance VALUES ('3','manisha','ABSENT','DEFAULT')")
#c.execute("INSERT INTO attendance NAME WHERE name='shailja'")
#c.execute("SELECT id,name FROM students INNER JOIN attendance ON students.id = attendance.id")
#c.execute("SELECT * FROM attendance WHERE NAME= 'monika'")
#c.execute("SELECT * FROM students")
#print (c.fetchone());
print (c.fetchall());
print ("Table created successfully");
conn.commit()
conn.close()

                                               
