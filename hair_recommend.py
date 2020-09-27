import os
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from keras.models import Sequential, load_model
from keras.preprocessing import image
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
classes = ['ショート', 'パーマ', 'マッシュ', 'ボウズ', 'ツーブロック', 'ロング']
num_classes = len(classes)
image_size = 64
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def get_face(img):
    # 画像をグレースケールへ変換
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # カスケードファイルのパス
    cascade_path = "haarcascade_frontalface_alt.xml"
    # カスケード分類器の特徴量取得
    cascade = cv2.CascadeClassifier(cascade_path)
    # 顔認識
    faces=cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1, minSize=(10,10))
    # 顔認識出来なかった場合
    if len(faces) == 0:
        face = faces
    # 顔認識出来た場合
    else:
        # 顔部分画像を取得
        for x,y,w,h in faces:
            face = img[y:y+h, x:x+w]
        # リサイズ
        face = cv2.resize(face, (image_size, image_size))
    pil_crop = Image.fromarray(face[:, :, ::-1])#Opencv to PIL
    return pil_crop
model = load_model('./model.h5')#学習済みモデルをロードする
graph = tf.get_default_graph()
@app.route('/', methods=['GET', 'POST'])
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
                pred_answer = "これは " + classes[predicted] + " です"
                return render_template("index.html",answer=pred_answer)
        return render_template("index.html",answer="")
if __name__ == "__main__":
    #deployするときはこっち
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)
    #ローカルで動かすときはこっちを使ってください
    #app.run()





