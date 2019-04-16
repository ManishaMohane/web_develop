import sqlite3
conn = sqlite3.connect('test1.db')
c = conn.cursor()
print ("Opened database successfully");
#c.execute("DROP TABLE attendance")
#c.execute('''CREATE TABLE attendance 
 #     (ID INTEGER PRIMARY KEY NOT NULL,
  #     NAME VARCHAR  NOT NULL,
   #    ATTENDANCE  VARCHAR  NOT NULL,
    #   Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
     #         );''')
c.execute("INSERT INTO attendance VALUES (1,'shailja','absent','DATETIME('now')' ")
c.execute("INSERT INTO attendance VALUES (2,'monika','absent','DATETIME('now')' ")
c.execute("INSERT INTO attendance VALUES (3,'manisha','absent','DATETIME('now')' ")
print ("Table created successfully");
conn.commit()
conn.close()

