import pymysql
from tkinter import *
from datetime import datetime




conn = pymysql.connect(host="15.164.212.236",port=3306, user='user',password='1234',db='foodmenu', charset='utf8')
curs = conn.cursor()
try:
	sqlQuery = 'CREATE TABLE login_details(shopname varchar(100), phone varchar(100), username varchar(100) primary key, password varchar(200), area varchar(100), created datetime);'
	curs.execute(sqlQuery)
	conn.commit()
except:
	pass




class LoginPage():
	def __init__(self, master,shop):
		self.master = master
		self.shop = shop
		master.label_username = Label(master, text="Username")
		master.entry_username = Entry(master)
		master.label_username.grid(row=0,column=0,sticky=E)
		master.entry_username.grid(row=0,column=1)
		
		master.label_password = Label(master, text="Password")
		master.entry_password = Entry(master, show="•")
		master.label_password.grid(row=1,column=0,sticky=E)
		master.entry_password.grid(row=1,column=1)
		
		login_button = Button(master,text="Login",command=lambda:self.validate(master.entry_username.get(),master.entry_password.get()))
		login_button.grid(columnspan=2)
		
		register_button = Button(master,text="Register",command=self.register)
		register_button.grid(columnspan=2)
	
	def validate(self,username,passwordB):
		sqlQuery = "SELECT password FROM login_details WHERE username="+repr(username)+";"
		curs.execute(sqlQuery)
		passwordA = curs.fetchone()
		sqlQuery = "SELECT shopname FROM login_details WHERE username="+repr(username)+";"
		curs.execute(sqlQuery)
		shopname = curs.fetchone()
		if not passwordA:
			print("Invalid Username")
		elif passwordA[0] != passwordB:
			print("Invalid Password")
		else:
			print("Login Successful")
			self.shop.name = shopname[0]
			conn.close()
			self.master.destroy()
			self.shop.enter()
	
	def reg(self):
		username = self.master.entry_username.get()
		shopname = self.master.entry_shopname.get()
		password = self.master.entry_password.get()
		area = self.master.entry_area.get()
		phone = self.master.entry_phone.get()
		timedate = str(datetime.now())
		sqlQuery = "INSERT INTO login_details VALUES("+repr(username)+", "+repr(shopname)+", "+repr(password)+", "+repr(area)+","+repr(timedate)+","+repr(phone)+");"
		curs.execute(sqlQuery)
		shopname = shopname.replace(" ","_")
		sqlQuery = "CREATE TABLE "+shopname+"_menu(food varchar(20) primary key, price int, ranking int);"
		curs.execute(sqlQuery)
		conn.commit()
		sqlQuery = "CREATE TABLE "+shopname+"_order(date_ datetime, food varchar(20), count int);"
		curs.execute(sqlQuery)
		conn.commit()
		self.shop.name = shopname
		print("Registration Successful")
		conn.close()
		self.master.destroy()
		self.shop.enter()

	def register(self):		
		for widget in self.master.winfo_children():
			widget.destroy()
		self.master.label_username = Label(self.master, text="Username")
		self.master.entry_username = Entry(self.master)
		self.master.label_username.grid(row=0,column=0,sticky=E)
		self.master.entry_username.grid(row=0,column=1)
		
		self.master.label_shopname = Label(self.master, text="Shopname")
		self.master.entry_shopname = Entry(self.master)
		self.master.label_shopname.grid(row=1,column=0,sticky=E)
		self.master.entry_shopname.grid(row=1,column=1)
		
		self.master.label_password = Label(self.master, text="Password")
		self.master.entry_password = Entry(self.master)
		self.master.label_password.grid(row=2,column=0,sticky=E)
		self.master.entry_password.grid(row=2,column=1)

		self.master.label_area = Label(self.master, text="area")
		self.master.entry_area = Entry(self.master)
		self.master.label_area.grid(row=3,column=0,sticky=E)
		self.master.entry_area.grid(row=3,column=1)

		self.master.label_phone = Label(self.master, text="phone")
		self.master.entry_phone = Entry(self.master)
		self.master.label_phone.grid(row=4,column=0,sticky=E)
		self.master.entry_phone.grid(row=4,column=1)
		
		register_button = Button(self.master,text="register",command=self.reg)
		register_button.grid(columnspan=2)

