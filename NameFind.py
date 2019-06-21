import os
import cv2
import pymysql
import datetime

def add_date(date):
	db = pymysql.connect("localhost","root","","Autobot" )
	cursor = db.cursor()
	cursor.execute("ALTER TABLE `attendance` ADD COLUMN IF NOT EXISTS `"+date+"` INT NOT NULL DEFAULT '0'")
	db.close()

def take_attendance(folder_no):
	now=datetime.datetime.now()
	date=now.strftime("%Y-%m-%d")
	add_date(date)
	db = pymysql.connect("localhost","root","","Autobot" )
	cursor = db.cursor()
	cursor.execute("UPDATE `attendance` SET `"+date+"` = '1' WHERE `attendance`.`folder_no`='"+str(folder_no)+"'")
	db.commit()
	cursor.execute("SELECT `Name` FROM `student_record` WHERE `folder_no`="+str(folder_no))
	data=cursor.fetchall()
	db.close()
	info='Name={0}'.format(*data[0])
	return info
	
def take_name(folder_no):
	db = pymysql.connect("localhost","root","","Autobot" )
	cursor = db.cursor()
	cursor.execute("SELECT * FROM `student_record` WHERE `folder_no`="+str(folder_no))
	data=cursor.fetchall()
	db.close()
	info='Roll NO={1}\tName={2}'.format(*data[0])
	return info
	
def AddName():
	db = pymysql.connect("localhost","root","","Autobot" )
	cursor = db.cursor()
	cursor.execute("SELECT COUNT(*) FROM `student_record`")
	data=cursor.fetchall()
	ID=data[0][0]
	ID+=1
	print("Write Your Details Below".center(25))
	Rno=input("Your Roll No 	")
	name=input("Your Name 	")
	f_name=input("Your Father's Name 	")
	branch=input("Your Branch ex:CSE 	")
	year=input("Your Year 	")
	email=input("Your email 	")
	psw=input("Your password 	")
	mobile=input("Your Mobile  ")
	sql="INSERT INTO `student_record` VALUES ("+str(ID)+","+str(Rno)+",'"+name+"','"+f_name+"','"+branch+"',"+str(year)+",'"+email+"','"+psw+"',"+str(mobile)+")"
	print(sql)
	cursor.execute(sql)
	db.commit()
	cursor.execute("INSERT INTO `attendance`(`folder_no`, `Roll_no`) VALUES ("+str(ID)+","+Rno+")")
	db.commit()
	db.close()
	print("Your  entry is Done in database.")
	directory='Newdataset/'+str(ID)
	try:
		if not os.path.exists(directory):
			os.makedirs(directory)
		else:
			print("New Data Entry Started.")	
		print("Entry successfully Done. Wait For image capturing..")
	except OSError:
		print ('Error: Creating directory. ' +  directory)
	return ID