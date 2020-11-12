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
    pil_crop = Image.fromarray(face[:, :, ::-1])#Opencv to PIL
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
        self.two_block_c = 0

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    hair_name = db.Column(db.String(100) )
    counts = db.Column(db.Integer)

    def __init__(self, hair_name,counts):
        self.hair_name = hair_name
        self.counts = counts

def print_all_history(session):
    hairs = History.query.all()
    total = 0
    for hair in hairs:
        total += hair.counts
        print(hair.id ,hair.hair_name ,hair.counts)
    return total    

model = load_model('./model.h5')#学習済みモデルをロードする
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
            session['logged_in']=True
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
                 #エラー出る可能性あります
                hit_hair_info = History.query.filter_by(hair_name=classes[predicted]).first()
                hit_hair_info.counts+=1 
                total = print_all_history(session)
                db.session.commit()

                pred_answer = "あなたのおすすめは " + classes[predicted] + " です"
                hairs = History.query.all()
                comment = '合計{}回'.format(str(total))
                comment_all = 'ボウズ{}回、ロング{}回、マッシュ{}回、パーマ{}回、ツーブロック{}回'.format(str(hairs[0].counts),
                                                                                                       str(hairs[1].counts),
                                                                                                       str(hairs[2].counts),
                                                                                                       str(hairs[3].counts),
                                                                                                       str(hairs[4].counts))
                return render_template("index.html",answer=pred_answer, comment=comment, allcomment=comment_all)
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
