id=input("enter your id")
name=input("enter your name") 
def insertOrUpdate(id,name):
      conn=sqlite3.connect("test1.db")
      cmd= "SELECT * FROM attendance WHERE ID="+str(id)
      cursor=conn.execute(cmd)
      isRecordExist=0
      for row in cursor:
        isRecordExist=1
      if(isRecordExist==1):
        cmd="UPDATE attendance SET NAME="+str(name)+"WHERE ID="+str(id)
      else:
        cmd="INSERT INTO attendance(id,Name) Values("+str(id)+","+str(name)+")"
      conn.execute(cmd)
      conn.commit()
      conn.close()
