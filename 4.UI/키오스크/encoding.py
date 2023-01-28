import pandas as pd
import numpy as np
import pymysql
df=pd.read_csv("국수나무_menu.csv")
df.to_csv("국수나무_menu_encoding.csv", encoding="utf-8", index=False)
shopname='철우네_인덕원점'
        sql_2 = "CREATE TABLE "+shopname+"_menu(food varchar(20) primary key, price int, ranking int);"
        sql_3 = "CREATE TABLE "+shopname+"_order(date_ datetime, food varchar(20), count int, order_no int);"
        sql_4 = "CREATE TABLE "+shopname+"_order_B(date_ datetime, food varchar(20), count int);"
        conn = pymysql.connect(host="database.ciylu0ctczrx.ap-northeast-2.rds.amazonaws.com",port=3306, user='root',password='pass123#',db='foodmenu', charset='utf8')
curs = conn.cursor()
            curs.execute(sql_2)
            curs.execute(sql_3)
            curs.execute(sql_4)
sql_5 = "select * from 국수나무_order a inner join 국수나무_menu b on a.food=b.food"
curs.execute(sql_5)
result= curs.fetchall()
pd.DataFrame(result)
