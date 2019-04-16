import sqlite3

def insertOrUpdate(ID, name):
    #connecting to the db
    conn =sqlite3.connect("test1.db")

    #check if id already exists
    query = "SELECT * FROM attendance WHERE ID="+str(ID)
    print (query)
    #returning the data in rows
    cursor = conn.execute(query)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if isRecordExist==1:
        cmd="UPDATE attendance SET Name=' "+str(name)+" ' WHERE ID="+str(ID) 
    else:
        cmd="INSERT INTO students(ID,Name) Values("+str(ID)+",' "+str(name)+" ' )"
    conn.execute(query)
    conn.commit()
    conn.close()
ID = input('Enter user id: ')
name = input('Enter name: ')
insertOrUpdate(ID, name)
