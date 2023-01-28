from flask import Flask, render_template, Response, request, session, redirect, url_for, make_response
from multiprocessing.connection import wait
import mediapipe as mp
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime
from collections import Counter
import pymysql, os, pyttsx3, cv2, sys, webbrowser, socket, time, json, boto3, shutil
from threading import Timer

# conn = pymysql.connect(host="localhost",user="root",password="Root123#",db="foodmenu")
# conn = pymysql.connect(host="database.ciylu0ctczrx.ap-northeast-2.rds.amazonaws.com",port=3306, user='root',password='pass123#',db='foodmenu', charset='utf8')
conn = pymysql.connect(host="base.ciylu0ctczrx.ap-northeast-2.rds.amazonaws.com",port=3306, user='root',password='pass123#',db='foodmenu', charset='utf8')
curs = conn.cursor()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)


if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    app = Flask(__name__, template_folder=template_folder)
else:
    app = Flask(__name__)

app = Flask(__name__)
app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.secret_key = "My_Key"

now = datetime.now()

df=pd.read_csv('./models/actions_list_vbell.csv')

actions=list(df.label.astype('str'))
print(actions)


path='./dataset'
try: 
    os.makedirs(path) 
except OSError: 
    if not os.path.isdir(path): 
        raise
file_list = os.listdir(path)

seq_length = 30
secs_for_action = 30

model = load_model('./models/models_vbell.h5')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


seq = []
action_seq = []
print_action=[]
last_action = None
order_num = [] 
shopname = None
menu = None
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

@app.route('/realtimeorders')
def RealTime_data():
    try:
        conn = pymysql.connect(host="base.ciylu0ctczrx.ap-northeast-2.rds.amazonaws.com",port=3306, user='root',password='pass123#',db='foodmenu', charset='utf8')
        curs = conn.cursor()
        sql="select * from "+shopname+"_order;"
        # Create a PHP array and echo it as JSON
        curs.execute(sql)
        conn.commit()
        df=pd.DataFrame(curs.fetchall())
        df.columns=['date_','food','count','order_no']
        df['date_']=df.date_.map(lambda x: x.strftime('%Y-%m-%d'))
        df.drop(columns="date_")
        order=[]
        global menu
        menu=dict()
        n=1
        for i in set(df.order_no):
            order.append(df[df.order_no==i].food.to_list())
            order.append(i)
            menu[str(n)]=order
            n+=1
            order=[]

        response = make_response(json.dumps(menu))
        response.content_type = 'application/json'
        conn.close()
        return response
    except:
        pass

def gen_frames():  # generate frame by frame from camera
    while cap.isOpened():
        ret, img = cap.read()
        #img0 = img.copy()

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                # Compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                v = v2 - v1 # [20, 3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) # Convert radian to degree

                d = np.concatenate([joint.flatten(), angle])

                seq.append(d)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                if len(seq) < seq_length:
                    continue

                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
    
                y_pred = model.predict(input_data).squeeze()

                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < 0.9:
                    continue

                action = actions[i_pred]
                action_seq.append(action)

                if len(action_seq) < 3:
                    continue

            
                global print_action
                global order_num
                this_action = '?'
                if action_seq[-1] == action_seq[-2] == action_seq[-3] :
                    this_action = action
                    print_action.append(this_action)

                cv2.putText(img, f'{this_action}', (int(img.shape[1]/10), int(img.shape[0]/2)), 4, fontScale=3.2, color=(128,0,255), thickness=5)
                cv2.putText(img, f'{order_num}', (25,(img.shape[0]-25)) , 4, fontScale=2.5, color=(128,0,255), thickness=4)
                if len(print_action) < 20:
                    continue
                
                this_action = Counter(print_action).most_common(1)[0][0]
                print(this_action)
                time.sleep(0.7)
                print_action=[]
                

                try: 
                    int(this_action)
                    order_num.append(this_action)
                    print(order_num)
                except:
                    if this_action == 'Wait':
                        time.sleep(0.7)
                    #10: commit - 두 손가락 구부리기
                    if this_action == 'Commit':
                        order="".join(order_num)
                        print(order)
                        print(menu)
                        try:
                            print(menu[str(order)][1])
                            conn = pymysql.connect(host="base.ciylu0ctczrx.ap-northeast-2.rds.amazonaws.com",port=3306, user='root',password='pass123#',db='foodmenu', charset='utf8')
                            curs = conn.cursor()
                            sqlQuery = "delete from "+shopname+"_order where order_no="+str(menu[str(order)][1])+";"
                            curs.execute(sqlQuery)
                            conn.commit()   
                            conn.close()                         
                        except:
                            pass
                        order_num = []
                        #11: backspace - 좌에서 우로 두 손가락 까딱
                    elif this_action == 'Backspace':
                        try:
                            order_num.pop()
                            print(order_num)
                        except:
                            pass
                        #12: call - 오케이
                    elif this_action == "Call":
                        order="".join(order_num)
                        print(order)
                        try:
                            num=order 
                            data ="번 손님 주문하신 음식 나왔습니다"
                            s = pyttsx3.init()
                            s.setProperty('rate',130)  
                            s.say(str(menu[str(num)][1])+data) 
                            s.runAndWait() 
                        except:
                            pass
                with open('./models/action.txt', 'a', newline='') as f: 
                    f.write(f"{now.year}-{now.month}-{now.day}-{now.hour}:{now.minute}:{now.second} | "+ "action: " +str(this_action)+"\n")
                f.close

            cv2.imshow('img', img)


            if (not ret):

                break

            else:

                ret, buffer = cv2.imencode('.jpg', img)

                frame = buffer.tobytes()

                yield (b'--frame\r\n'

                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                       # concat frame one by one and show result
    cap.release()





@app.route("/",  methods=['POST','GET'])
@app.route("/index",  methods=['POST','GET'])
def index():
    conn = pymysql.connect(host="base.ciylu0ctczrx.ap-northeast-2.rds.amazonaws.com",port=3306, user='root',password='pass123#',db='foodmenu', charset='utf8')
    curs = conn.cursor()
    if request.method == "POST":
        global shopname
        username = request.form.get('username')
        password = request.form.get('password')
        shopname=username.split('_')[0]
        sql = "SELECT shopname FROM login_details WHERE username = '" +username+ "' and password = '"+password+"';"
        if curs.execute(sql) > 0:
            conn.commit()
            session['username'] = request.form['username']
            return redirect(url_for('index_after'))
        else:
            return redirect(url_for('index'))
    conn.close()
    return render_template("index.html")
 

@app.route('/video_feed')

def video_feed():

    """Video streaming route. Put this in the src attribute of an img tag."""

    return Response(gen_frames(),

                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/handmotion", methods=['GET','POST'])

def handmotion():
    if 'username' in session:
        username = session['username']
        print(username)
    return render_template('handmotion.html')

@app.route("/index_after", methods=['GET','POST'])

def index_after():
    if 'username' in session:
        username = session['username']
        print(username)
    return render_template('index_after.html')


@app.route("/signup", methods=['GET','POST'])

def signup():
    conn = pymysql.connect(host="base.ciylu0ctczrx.ap-northeast-2.rds.amazonaws.com",port=3306, user='root',password='pass123#',db='foodmenu', charset='utf8')
    curs = conn.cursor()
    if request.method == "POST":
        username = request.form.get('username')
        shopname = request.form.get('shopname')
        password = request.form.get('password')
        phone = request.form.get('phone')
        area = request.form.get('area')
        passwordcheck = request.form.get('passwordcheck')
        
        sql="INSERT INTO login_details VALUES('"+username+"', '"+shopname+"', '"+password+"', '"+area+"','"+str(now)+"','"+phone+"');"
        sql_2 = "CREATE TABLE "+shopname+"_menu(food varchar(20), price int, ranking int);"
        sql_3 = "CREATE TABLE "+shopname+"_order(date_ datetime, food varchar(20), count int, order_no int);"
        sql_4 = "CREATE TABLE "+shopname+"_order_B(date_ datetime, food varchar(20), count int);"
        if str(passwordcheck) != str(password):
            if curs.execute(sql)> 0:
                curs.execute(sql_2)
                curs.execute(sql_3)
                curs.execute(sql_4)
                conn.commit()
                return redirect(url_for('index'))
        else:
            return '저장에 문제가 생겼습니다.'
    conn.close()
    return render_template('signup.html')

@app.route("/option", methods=['GET','POST'])

def option():
    if 'username' in session:
        username = session['username']
        print(username)

    return render_template('option.html')


@app.route('/live-data')
def live_data():
    conn = pymysql.connect(host="base.ciylu0ctczrx.ap-northeast-2.rds.amazonaws.com",port=3306, user='root',password='pass123#',db='foodmenu', charset='utf8')
    curs = conn.cursor()
    sql="select sum(order_price) from (select a.date_, a.food, count, count*price as order_price from "+shopname+"_order_B a LEFT join "+shopname+"_menu b on a.food=b.food union select a.date_, a.food, count, count*price as order_price from "+shopname+"_order_B a RIGHT join "+shopname+"_menu b on a.food=b.food) d;"
    # Create a PHP array and echo it as JSON
    curs.execute(sql)
    conn.commit()
    results=int(pd.DataFrame(curs.fetchall())[0][0])
    data = [time.time() * 1000, results]
    conn.commit()
    response = make_response(json.dumps(data))
    response.content_type = 'application/json'
    conn.close()
    return response

@app.route('/chart_realtime')
def chart_realtime():
    return render_template('chart_realtime.html')
#plotly Chart
# @app.route('/pie')
# def pie():
#     conn = pymysql.connect(host="base.ciylu0ctczrx.ap-northeast-2.rds.amazonaws.com",port=3306, user='root',password='pass123#',db='foodmenu', charset='utf8')
#     curs = conn.cursor()
#     sql="select a.date_, a.food, count, count*price as order_price from "+shopname+"_order_B a LEFT join "+shopname+"_menu b on a.food=b.food union select a.date_, a.food, count, count*price as order_price from "+shopname+"_order_B a RIGHT join "+shopname+"_menu b on a.food=b.food;"
#     curs.execute(sql)
#     conn.commit()
#     df=pd.DataFrame(curs.fetchall())
#     df.columns=['date_','food','count','price_all']
#     df = pd.DataFrame(df['food'].value_counts()).reset_index()
#     df.columns = ['food', 'counts']
#     go_fig = go.Figure()
#     obj = go.Pie(labels = df['food'].values.tolist() , values = df['counts'].values.tolist(), hole=.3)
#     go_fig.add_trace(obj)    
#     go_fig.write_html('./templates/pie.html')  
#     conn.close()
#     return render_template('pie.html')


@app.route('/pie')
def pie():
    conn = pymysql.connect(host="base.ciylu0ctczrx.ap-northeast-2.rds.amazonaws.com",port=3306, user='root',password='pass123#',db='foodmenu', charset='utf8')
    curs = conn.cursor()
    sql="select a.date_, a.food, count, count*price as order_price from "+shopname+"_order_B a LEFT join "+shopname+"_menu b on a.food=b.food union select a.date_, a.food, count, count*price as order_price from "+shopname+"_order_B a RIGHT join "+shopname+"_menu b on a.food=b.food;"
    curs.execute(sql)
    conn.commit()
    df=pd.DataFrame(curs.fetchall())
    df.columns=['date_','food','count','price_all']
    df = pd.DataFrame(df['food'].value_counts()).reset_index()
    df.columns = ['food', 'counts']
    foods=df['food'].values.tolist()
    values=df['counts'].values.tolist()
    conn.close()
    return render_template('pie.html', foods=foods, values=values)

@app.route('/chart_daytime')
def chart_daytime():
    return render_template('chart_daytime.html')


@app.route("/chart_call", methods=['GET','POST'])
def chart_call():
    conn = pymysql.connect(host="base.ciylu0ctczrx.ap-northeast-2.rds.amazonaws.com",port=3306, user='root',password='pass123#',db='foodmenu', charset='utf8')
    curs = conn.cursor()
    try:
        if 'username' in session:
            username = session['username']
            print(username)
        sql ='select a.date_, a.food, count, count*price as order_price from '+shopname+'_order_B a LEFT join '+shopname+'_menu b on a.food=b.food union select a.date_, a.food, count, count*price as order_price from '+shopname+'_order_B a RIGHT join '+shopname+'_menu b on a.food=b.food;'
        curs.execute(sql)
        conn.commit()
        df=pd.DataFrame(curs.fetchall())[:-1]
        df.columns =['date_', 'food','count','all_price']
        df.to_csv(shopname+'.csv', index= False)
        s3 = boto3.client('s3')
        s3.upload_file(shopname+'.csv','chunsik', shopname+'/'+shopname+str(datetime.now())+'.csv')
    except:
        pass
    conn.close()
    return render_template('option.html')
                              
@app.route('/order_wait')
def order_wait():
    return render_template('order_wait.html')


@app.route('/bar')
def bar():
    conn = pymysql.connect(host="base.ciylu0ctczrx.ap-northeast-2.rds.amazonaws.com",port=3306, user='root',password='pass123#',db='foodmenu', charset='utf8')
    curs = conn.cursor()
    sql="select a.date_, a.food, count, count*price as order_price from "+shopname+"_order_B a LEFT join "+shopname+"_menu b on a.food=b.food union select a.date_, a.food, count, count*price as order_price from "+shopname+"_order_B a RIGHT join "+shopname+"_menu b on a.food=b.food;"
    curs.execute(sql)
    conn.commit()
    df=pd.DataFrame(curs.fetchall())[:-1]
    df.columns=['date_','food','count','price_all']
    df=df[:-1]
    df['date_']=df.date_.map(lambda x: x.strftime('%Y.%m'))
    df=df.groupby('date_').sum().reset_index()
    y_preds = df['price_all'].tolist()
    days = df['date_'].tolist()
    return render_template('bar.html', days=days, y_preds=y_preds)

@app.route('/chart_monthtime')
def chart_monthtime():
    return render_template('chart_monthtime.html')

actions_labels = []
actions_num = []
actions_add = []

@app.route("/recording", methods=['GET','POST'])
def recording():
    global actions_labels
    global actions_num
    global actions_add
    if request.method == "POST":
        actions_labels = request.form.get('labels')
        actions_labels=actions_labels.split(',')
        actions_num=list(range(len(actions_labels)))
        actions_add=list(set(actions_num))
        actions_add.sort()
        return redirect(url_for('recording_after'))
    pd.DataFrame({'index':actions_num,'label':actions_labels}).set_index('index').to_csv('./dataset/actions_list.csv')
    return render_template('recording.html')



def get_frames(): 
    while cap.isOpened():
        for idx, action in enumerate(actions_add):
            print(idx, action)
            data = []
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            cv2.imshow('img', img)
            cv2.waitKey(3000)

            start_time = time.time()

            while time.time() - start_time < secs_for_action:
                ret, img = cap.read()

                img = cv2.flip(img, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if result.multi_hand_landmarks is not None:
                    for res in result.multi_hand_landmarks:
                        joint = np.zeros((21, 4))
                        for j, lm in enumerate(res.landmark):
                            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                        # Compute angles between joints
                        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                        v = v2 - v1 # [20, 3]
                        # Normalize v
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                        # Get angle using arcos of dot product
                        angle = np.arccos(np.einsum('nt,nt->n',
                            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                        angle = np.degrees(angle) # Convert radian to degree

                        angle_label = np.array([angle], dtype=np.float32)
                        angle_label = np.append(angle_label, len(actions_add)+idx)
                        # 커스텀 동작 추가 시에 라이브 하고 윗 부분을 주석처리
                        #angle_label = np.append(angle_label, idx)
                        d = np.concatenate([joint.flatten(), angle_label])

                        data.append(d)

                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
                        cv2.putText(img, f'Waiting for collecting {action} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                cv2.imshow('img', img)
                
                if (not ret):

                    break

                else:

                    ret, buffer = cv2.imencode('.jpg', img)

                    frame = buffer.tobytes()

                    yield (b'--frame\r\n'

                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            data = np.array(data)
            print(action, data.shape)
            np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

            # Create sequence data
            full_seq_data = []
            for seq in range(len(data) - seq_length):
                full_seq_data.append(data[seq:seq + seq_length])

            full_seq_data = np.array(full_seq_data)
            print(action, full_seq_data.shape)
            np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
    cap.release()




@app.route('/video_record')
def video_record():
    return Response(get_frames(),

                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/recording_after", methods=['GET','POST'])
def recording_after():
    if request.method == "POST":
        path="./dataset"
        file_list = os.listdir(path)
        np_process_list=[]
        for i in file_list:
            if i.startswith('seq') == True:
                np_process_list.append(i)
            else: 
                continue
        s3 = boto3.client('s3')
        for i in np_process_list:
            s3.upload_file(path+"/"+i,'chunsik', shopname+'/'+'/mymotion'+str(datetime.now().strftime('%Y.%m.%d'))+'/'+i)
        shutil.rmtree(path)
        return redirect(url_for('option'))
    return render_template('recording_after.html')

port=5000
host_ip=socket.gethostbyname_ex(socket.gethostname())[2][0]
def open_browser():
      webbrowser.open_new('http://'+host_ip+':'+str(port)+'/') 
                  
if __name__ == "__main__":
    Timer(1,open_browser).start();
    app.run(host='0.0.0.0')  