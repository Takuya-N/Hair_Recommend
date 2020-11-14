import os
from flask import Flask, Response, request, redirect, url_for, render_template, flash, session
from werkzeug.utils import secure_filename
from keras.models import Sequential, load_model
from keras.preprocessing import image
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from keras.models import Sequential, load_model
from keras.preprocessing import image
import tensorflow as tf

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

classes = ['ボウズ', 'ロング', 'マッシュ', 'パーマ', 'ツーブロック']
num_classes = len(classes)
image_size = 64
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_face(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cascade_path = "haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(cascade_path)

    faces=cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1, minSize=(10,10))
    if len(faces) == 0:
        face = img
 
    else:
        for x,y,w,h in faces:
            face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (image_size, image_size))
    pil_crop = Image.fromarray(face[:, :, ::-1])
    return pil_crop

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    counts = db.Column(db.Integer)
    bouzu_c = db.Column(db.Integer)
    long_c = db.Column(db.Integer)
    mash_c = db.Column(db.Integer)
    pama_c = db.Column(db.Integer)
    two_block_c = db.Column(db.Integer)

    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.counts = 0
        self.bouzu_c = 0
        self.long_c = 0
        self.mash_c = 0
        self.pama_c = 0
        self.two_block_c = 0

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    hair_name = db.Column(db.String(100) )
    counts = db.Column(db.Integer)

    def __init__(self, hair_name,counts):
        self.hair_name = hair_name
        self.counts = 0


model = load_model('./model.h5')
graph = tf.get_default_graph()


@app.route('/register/',methods=['GET','POST'])
def register():
    if request.method=='POST':
        try:
            db.session.add(User(username=request.form['username'], password=request.form['password']))
            db.session.commit()
            return redirect(url_for(''))
        except:
            return render_template('register.html')
    else:
        return render_template('register.html')

@app.route('/', methods=['GET','POST'])
def login():
    if request.method == 'GET':
       return render_template('login.html')
    else:
        u = request.form['username']
        p = request.form['password']
        data = User.query.filter_by(username=u, password=p).first()
        if data is not None:
            session['logged_in'] = True
            session['user_name'] = u
            return redirect(url_for('upload_file'))
        return render_template('login.html', message="login_fail")    

@app.route('/index/', methods=['GET', 'POST'])
def upload_file():
    global graph
    with graph.as_default():
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('ファイルがありません')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('ファイルがありません')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(UPLOAD_FOLDER, filename))
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                #受け取った画像を読み込み、np形式に変換
                img = image.load_img(filepath, grayscale=False, target_size=(image_size,image_size))
                ocv_im = np.asarray(img)[:, :, ::-1]#PIL to opencv
                img = get_face(ocv_im)
                img = image.img_to_array(img)
                data = np.array([img])
                #変換したデータをモデルに渡して予測する
                result = model.predict(data)[0]
                predicted = result.argmax()
                #合計結果の処理
                total = 0
                hairs = History.query.all()
                hit_hair = History.query.filter_by(hair_name=classes[predicted]).first()
                hit_hair.counts += 1 
                for hair in hairs:
                    total += hair.counts
                db.session.commit()
                #ユーザーデータの処理
                user_data = User.query.filter_by(username =session['user_name']).first()
                user_data.counts += 1
                if predicted == 0:
                    user_data.bouzu_c += 1
                elif predicted == 1:
                    user_data.long_c += 1
                elif predicted == 2:
                    user_data.mash_c += 1
                elif predicted == 2:
                    user_data.pama_c += 1
                else:
                    user_data.two_block_c += 1
                db.session.commit()
                user_data = User.query.filter_by(username =session['user_name']).first()   
                #結果の表示
                pred_answer = '{}さんのおすすめは{}です'.format(user_data.username,classes[predicted])
                comment = '{}さんの合計{}回'.format(user_data.username,str(user_data.counts))
                comment_all = 'ボウズ{}回、ロング{}回、マッシュ{}回、パーマ{}回、ツーブロック{}回'.format(str(user_data.bouzu_c),
                                                                                                       str(user_data.long_c),
                                                                                                       str(user_data.mash_c),
                                                                                                       str(user_data.pama_c),
                                                                                                       str(user_data.two_block_c))
                comment2 = 'みんなの合計{}回'.format(str(total))
                comment_all2 = 'ボウズ{}%、ロング{}%、マッシュ{}%、パーマ{}%、ツーブロック{}%'.format(str(hairs[0].counts*100//total),
                                                                                                       str(hairs[1].counts*100//total),
                                                                                                       str(hairs[2].counts*100//total),
                                                                                                       str(hairs[3].counts*100//total),
                                                                                                       str(hairs[4].counts*100//total))
                
                return render_template("index.html",answer=pred_answer, comment=comment, allcomment=comment_all,comment2=comment2, allcomment2=comment_all2)
        return render_template("index.html",answer="")

@app.route('/logout', methods=['GET'])
def logout():
    session['logged_in'] = False
    return redirect(url_for('login'))

if(__name__ == "__main__"):
    app.secret_key = "ThisIsNotASecret:p"
    #DBを作成する

    db.create_all()
    #初期設定を行います
    for hair in classes:
        #既存のレコードがあるかを確認します
        check = History.query.filter_by(hair_name=hair).first()
        #もし、あればadd（作成）は行いません。
        #もし、なければaddを行います。
        #if文を使う
        if check :
            print("レコードはあります--> {} ".format(check.hair_name))
        else :
            print("レコードを作成しました--> {} ".format(hair))
            db.session.add(History(hair_name=hair, counts=0) )

        db.session.commit()
    #deployするときはこっち
    #port = int(os.environ.get('PORT', 8080))
    #app.run(host ='0.0.0.0',port = port)
    #ローカルで動かすときはこっちを使ってください
    app.run()