from tkinter import *
import foodmenu
import loginSystem
import pymysql
import operator
import datetime
import webbrowser
import socket
host=socket.gethostbyname(socket.gethostname())
# conn = pymysql.connect(host="localhost",user="root",password="root123",db="foodmenu")
conn = pymysql.connect(host="15.164.212.236",port=3306, user='user',password='1234',db='foodmenu')
curs = conn.cursor()



class Shop:
	def __init__(self, master):
		self.name = None
		self.master = master
		
		welcome_canvas = Canvas(master,width=360,height=140)
		welcome_canvas.pack()
		welcome_canvas.create_text(180,70,fill="slate blue", text="WELCOME",font=("Times",48,"bold"))
		
		enter_button = Button(master,text="Enter", command=self.enter)
		enter_button.config(width=360,pady=135,bd=5,fg="green",relief=GROOVE,font="SNAS 40")
		enter_button.pack()

		login_button = Button(master, text="Shopkeeper Login", command=self.login)
		login_button.config(width=360)
		login_button.pack()
	
		label_credits = Label(master,text="Icon made by Freepik from www.flaticon.com",fg="blue",cursor="hand2")
		label_credits.pack()
		label_credits.bind("<Button-1>",self.credit_link)
	
	def __del__(self):
		conn.close()
	
	def credit_link(self,event):
		webbrowser.open_new(r"https://www.flaticon.com/authors/freepik")
	
	def login(self):
		loginWin = Toplevel(self.master)
		loginPage = loginSystem.LoginPage(loginWin,self)
		
	def enter(self):
		for widget in self.master.winfo_children():
			widget.destroy()
		label_heading = Label(self.master,text="Menu")
		label_heading.pack()
		
		self.menuFrame = Frame(self.master)
		self.menuFrame.config(width=360,relief=GROOVE,bd=1)
		self.menuFrame.pack()
		
		if self.name != None:
			self.name = self.name.replace(" ","_")
			self.shopkeeper()
			self.getMenu()
			self.displayMenu()
		else:
			self.displayShops()


	def proceed(self,shopname):
		self.name = shopname.get()
		self.name = self.name.replace(" ","_")
		print("selected shop "+self.name+";")
		if self.name != None:
			for widget in self.shopFrame.winfo_children():
				widget.destroy()
			self.getMenu()
			self.interactiveMenu()
			self.customer()
		else:
			print("Please Select a Shop!")

	def proceed_str(self,shopname):
		self.name = shopname
		print("selected shop "+self.name+";")
		if self.name != None:
			for widget in self.shopFrame.winfo_children():
				widget.destroy()
			self.getMenu()
			self.interactiveMenu()
			self.customer()
		else:
			print("Please Select a Shop!")

	def displayShops(self):
		sqlQuery = "SELECT shopname FROM login_details;"
		self.shopFrame = Frame(self.master.master)
		self.shopFrame.pack()
		curs.execute(sqlQuery)
		shops = curs.fetchall()
		shopname = StringVar()
		shopname.set("My_Shop")
		for shop in shops:
			b = Radiobutton(self.shopFrame, text=shop[0], variable=shopname,value=shop[0],indicatoron=0)
			b.pack(fill=X)
		self.proceed_button = Button(self.shopFrame, text="Proceed", command=lambda:self.proceed(shopname))
		self.proceed_button.config(bg="green3")
		self.proceed_button.pack(side=BOTTOM)

	def getMenu(self):
		self.menu = foodmenu.Menu()
		sqlQuery = "SELECT * FROM "+self.name+"_menu;"
		curs.execute(sqlQuery)
		foods = curs.fetchall()
		for values in foods:
			food = foodmenu.Food(*values)
			self.menu.add(food)
		self.menu.reduceRanks()
	
	def clearEditFrame(self):
		for widget in self.editFrame.winfo_children():
			widget.destroy()
	
	def click_checkbox(self):
		for foodname in self.order.keys():
			if self.order[foodname].get() == 1 and self.quantity[foodname]==0:
				self.quantity[foodname] = 1
			elif self.order[foodname].get() == 0 and self.quantity[foodname] > 0:
				self.quantity[foodname] = 0
		self.interactiveOrder()
	
	def increment(self,foodname):
		if self.quantity[foodname] <= 10:
			self.quantity[foodname] += 1
			self.interactiveOrder()
		else:
			print("Only 10 of a food per order allowed!")
	
	def decrement(self,foodname):
		if self.quantity[foodname] >= 0:
			self.quantity[foodname] -= 1
			self.interactiveOrder()
		if self.quantity[foodname] == 0:
			self.order[foodname].set(0)
				
	def interactiveOrder(self):
		for widget in self.orderFrame.winfo_children():
			widget.destroy()
			
		label_name = Label(self.orderFrame,text="Food")
		label_quantity = Label(self.orderFrame,text="Quanitity")
		label_price = Label(self.orderFrame,text="Price")
		
		label_name.grid(row=0,column=0)
		label_quantity.grid(row=0,column=1)
		label_price.grid(row=0,column=2)
		
		self.total_cost = 0
		r = 1
		for foodname in self.order.keys():
			if self.quantity[foodname] > 0:
				food = self.menu.get(foodname)
				
				b = Checkbutton(self.orderFrame, text=food.name, variable=self.order[food.name],command=self.click_checkbox)
				b.grid(row=r,column=0,sticky=W)
				
				qnty = self.quantity[foodname]
				label_qnty = Label(self.orderFrame,text=repr(qnty))
				label_qnty.grid(row=r,column=1)
				
				cost = repr(food.price)+" X "+repr(qnty)+" = "+repr(food.price*qnty)
				label_cost = Label(self.orderFrame,text=cost)
				label_cost.grid(row=r,column=2)
				self.total_cost += food.price*qnty
				
				decr_button = Button(self.orderFrame,text="-",command=lambda foodname=foodname: self.decrement(foodname))
				decr_button.grid(row=r,column=3)
				
				incr_button = Button(self.orderFrame,text="+",command=lambda foodname=foodname: self.increment(foodname))
				incr_button.grid(row=r,column=4)
				
				r += 1
		label_total_cost = Label(self.orderFrame,text="Total Cost = "+repr(self.total_cost))
		label_total_cost.grid(row=r,column=0,columnspan=3)
		
		proceed_button = Button(self.orderFrame,text="Proceed",command=self.submit_order)
		proceed_button.grid(columnspan=5)
	
	def interactiveMenu(self):
		for widget in self.menuFrame.winfo_children():
			widget.destroy()
		label_name = Label(self.menuFrame,text="Food")
		label_price = Label(self.menuFrame,text="Price")
		label_name.grid(row=0,column=0)
		label_price.grid(row=0,column=1)
		self.order = {}
		self.quantity = {}
		r = 1
		for food in (sorted(self.menu.menu.values(), key=operator.attrgetter("ranking"),reverse=True)):
			self.order[food.name] = IntVar(0)
			self.quantity[food.name] = 0
			
			b = Checkbutton(self.menuFrame, text=food.name, variable=self.order[food.name],command=self.click_checkbox)
			b.grid(row=r,column=0,sticky=W)
			
			label_price = Label(self.menuFrame,text=repr(food.price))
			label_price.grid(row=r,column=1)
			r += 1
	
	def displayMenu(self):
		for widget in self.menuFrame.winfo_children():
			widget.destroy()
		label_name = Label(self.menuFrame,text="Food")
		label_price = Label(self.menuFrame,text="Price")
		label_name.grid(row=0,column=0)
		label_price.grid(row=0,column=1)
		r = 1
		for food in (sorted(self.menu.menu.values(), key=operator.attrgetter("ranking"),reverse=True)):
			label_name = Label(self.menuFrame,text=food.name)
			label_price = Label(self.menuFrame,text=repr(food.price))
			label_name.grid(row=r,column=0)
			label_price.grid(row=r,column=1)
			r += 1

	def submit_order(self):
		for widget in self.master.winfo_children():
			widget.destroy()
		timedate=str(datetime.datetime.now())
		shopname_canvas = Canvas(self.master,width=360,height=140)
		shopname_canvas.pack()
		shopname = self.name.replace("_"," ")
		shopname_canvas.create_text(180,70,fill="royal blue", text=shopname,font=("Times",50-len(shopname),"bold"))
		
		self.orderFrame = Frame(self.master,relief=GROOVE)
		self.orderFrame.pack()
		label_name = Label(self.orderFrame,text="Food")
		label_quantity = Label(self.orderFrame,text="Quanitity")
		label_price = Label(self.orderFrame,text="Net Amount")
		
		label_name.grid(row=0,column=0)
		label_quantity.grid(row=0,column=1)
		label_price.grid(row=0,column=2)
		
		r=1
		
		for foodname in self.order.keys():
			food = self.menu.get(foodname)
			qnty = self.quantity[foodname]
			if qnty > 0:
				
				label_foodname = Label(self.orderFrame,text=foodname)
				label_foodname.grid(row=r,column=0)
				
				label_qnty = Label(self.orderFrame,text=repr(qnty))
				label_qnty.grid(row=r,column=1)
				
				cost = repr(food.price)+" X "+repr(qnty)+" = "+repr(food.price*qnty)+" Rs"
				label_cost = Label(self.orderFrame,text=cost)
				label_cost.grid(row=r,column=2)
				
				r += 1
			
			sqlQuery = "UPDATE "+self.name+"_menu SET ranking="+repr(food.ranking+qnty)+" WHERE food="+repr(food.name)+";"
			curs.execute(sqlQuery)
			conn.commit()
			sqlQuery = "INSERT INTO "+self.name+"_order VALUES ("+repr(timedate)+", "+repr(food.name)+","+repr(qnty)+");"
		#	order(menu_no int  primary key, food varchar(20), price int, ranking int);?
			curs.execute(sqlQuery)
			conn.commit()
			sqlQuery = "DELETE FROM "+self.name+"_order WHERE count = 0;"
			curs.execute(sqlQuery)
			conn.commit()			
		
		detailsFrame = Frame(self.master,pady=10)
		detailsFrame.pack()
		
		label_total_cost = Label(detailsFrame,text="Total Amount to pay = "+repr(self.total_cost))
		label_total_cost.grid(row=r,column=0,columnspan=3)
		
		now = datetime.datetime.now()
		label_datetime = Label(detailsFrame,text=now.strftime("%d-%m-%Y   %H:%M"))
		label_datetime.grid(row=r+1,column=0,columnspan=3)

		self.Return_button = Button(self.shopFrame, text="Return_button", command=lambda:self.proceed_str(shopname))
		self.Return_button.config(bg="green3")
		self.Return_button.pack(side=BOTTOM)		
			
	def click_add(self):
		name = self.entry_foodname.get()
		price = float(self.entry_foodprice.get())
		ranking = 0
		food = foodmenu.Food(name,price,ranking)
		self.menu.add(food)
		self.displayMenu()
		
	def add(self,master):
		self.clearEditFrame()
		label_foodname = Label(master,text="Name")
		label_foodname.grid(row=0, sticky=W, padx=10)
		self.entry_foodname = Entry(master)
		self.entry_foodname.grid(row=0, column=1, sticky=E, padx=15, pady=10)
		
		label_foodprice = Label(master,text="price")
		label_foodprice.grid(row=1, sticky=W, padx=10)
		self.entry_foodprice = Entry(master)
		self.entry_foodprice.grid(row=1, column=1, sticky=E, padx=15, pady=10)
		
		add_button = Button(master,text="Add",command=self.click_add)
		add_button.grid(row=2,columnspan=2)

	def click_price(self):
		name = self.entry_foodname.get()
		newPrice = float(self.entry_foodprice.get())
		self.menu.changePrice(name,newPrice)
		self.displayMenu()
		
	def changePrice(self,master):
		self.clearEditFrame()
		label_foodname = Label(master,text="Name")
		label_foodname.grid(row=0, sticky=W, padx=10)
		self.entry_foodname = Entry(master)
		self.entry_foodname.grid(row=0, column=1, sticky=E, padx=15, pady=10)
		
		label_foodprice = Label(master,text="New price")
		label_foodprice.grid(row=1, sticky=W, padx=10)
		self.entry_foodprice = Entry(master)
		self.entry_foodprice.grid(row=1, column=1, sticky=E, padx=15, pady=10)
		
		change_button = Button(master,text="Change",command=self.click_price)
		change_button.grid(row=2,columnspan=2)	

	def click_name(self):
		name = self.entry_foodname.get()
		newName = self.entry_newName.get()
		self.menu.changeName(name,newName)
		self.displayMenu()
		
	def changeName(self,master):
		self.clearEditFrame()
		label_foodname = Label(master,text="Old Name")
		label_foodname.grid(row=0, sticky=W, padx=10)
		self.entry_foodname = Entry(master)
		self.entry_foodname.grid(row=0, column=1, sticky=E, padx=15, pady=10)
		
		label_newName = Label(master,text="New Name")
		label_newName.grid(row=1, sticky=W, padx=10)
		self.entry_newName = Entry(master)
		self.entry_newName.grid(row=1, column=1, sticky=E, padx=15, pady=10)
		
		change_button = Button(master,text="Change",command=self.click_name)
		change_button.grid(row=2,columnspan=2)	
	
	def click_remove(self):
		name = self.entry_foodname.get()
		self.menu.remove(name)
		self.displayMenu()
		
	def remove(self,master):
		self.clearEditFrame()
		
		label_foodname = Label(master,text="Name")
		label_foodname.grid(row=0, sticky=W, padx=10)
		self.entry_foodname = Entry(master)
		self.entry_foodname.grid(row=0, column=1, sticky=E, padx=15, pady=10)
		
		remove_button = Button(master,text="Remove",command=self.click_remove)
		remove_button.grid(row=1,columnspan=2)
	
	def click_clear(self):
		self.menu.clearAll()
		self.displayMenu()
		
	def clearAll(self,master):
		self.clearEditFrame()
		
		clear_button = Button(master,text="Clear All",command=self.click_clear)
		clear_button.grid(row=1,columnspan=2)

	def shopkeeper(self):
		self.master.master.title(self.name)
		
		self.editFrame = Frame(self.master,pady=10)
		self.editFrame.pack()
		self.editFrame.config(width=360)
		
		menubar = Menu(self.master)
		self.master.master.config(menu=menubar)
		
		newmenu = Menu(menubar,tearoff=0)
		newmenu.add_command(label="Add",command=lambda:self.add(self.editFrame))
		menubar.add_cascade(label="New",menu=newmenu)
		
		editmenu = Menu(menubar,tearoff=0)
		editmenu.add_command(label="Price",command=lambda:self.changePrice(self.editFrame))
		editmenu.add_command(label="Name",command=lambda:self.changeName(self.editFrame))
		menubar.add_cascade(label="Edit",menu=editmenu)
		
		removemenu = Menu(menubar,tearoff=0)
		removemenu.add_command(label="One",command=lambda:self.remove(self.editFrame))
		removemenu.add_command(label="All",command=lambda:self.clearAll(self.editFrame))
		menubar.add_cascade(label="Remove",menu=removemenu)
		
		savemenu = Menu(menubar,tearoff=0)
		savemenu.add_command(label="Save Menu",command=lambda:self.savemenu())
		menubar.add_cascade(label="Save",menu=savemenu)
	
	def customer(self):
		self.master.master.title(self.name)
		
		self.orderFrame = Frame(self.master,relief=GROOVE,pady=10)
		self.orderFrame.config(width=360)
		self.orderFrame.pack()
		
		label_test = Label(self.orderFrame,text="No Orders Yet")
		label_test.pack()
		
	def savemenu(self):
		sqlQuery = "DELETE FROM "+self.name+"_menu"+";"
		curs.execute(sqlQuery)
		for food in self.menu.menu.values():
			sqlQuery = "INSERT INTO "+self.name+"_menu VALUES("+repr(food.name)+", "+repr(food.price)+", "+repr(food.ranking)+");" 
			curs.execute(sqlQuery)
		conn.commit()
		print("Menu Updated")	
	
root = Tk()
root.geometry("480x740")
root.title("Shop")
root.iconbitmap("dish.ico")
mainFrame = Frame(root)
mainFrame.pack()
shopWindow = Shop(mainFrame)
root.mainloop()