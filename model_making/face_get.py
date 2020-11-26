import os
import cv2
import glob

path_bouzu =[]
path_long = []
path_mash = []
path_pama = []
path_two_block = []

hair_list=['bouzu','long','mash','pama','two_block']
path_list=[path_bouzu,path_long,path_mash,path_pama,path_two_block]

for i in range(len(hair_list)):
    path = './'+hair_list[i]
    path_list[i] = glob.glob(path +'/*')

cascade_path = 'haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_path)

for i in range(len(path_list)):
    number_face = 0
    for img_path in path_list[i]:
        faces = 0
        base_name = os.path.basename(img_path)
        name, ext = os.path.splitext(base_name)
        img_src = cv2.imread(img_path, 1)
        img_gray = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=1, minSize=(10,10))
        if len(faces) == 0:
            continue
        else:
            for x,y,w,h in faces:
                face = img_src[y:y+h, x:x+w]
            face = cv2.resize(face, (64,64))
            dirname = hair_list[i] + '_face'
            if not os.path.exists(dirname):
                os.mkdir(dirname)
         
            file_name = dirname + '_' + str(number_face) + ext 
            file_path = os.path.join(dirname, file_name)   
            number_face += 1
            cv2.imwrite(file_path, face)





