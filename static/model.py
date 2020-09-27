from keras.layers import Dense, Dropout, Flatten, Input
from keras.applications.vgg16 import  VGG16
from keras.models import Model, Sequential
from keras import optimizers

hair_list = ['short','two_block', 'long', 'pama', 'mash', 'bouzu']
def get_path_hair(hair):
    hair_path = glob.glob('./'+ hair + '')
    return hair_path

def get_img_hair(hair):
    hair_path = get_path_hair(hair):
    img_hair = []
    for i in range(len(hair_path)):
        img = cv2.imread(hair_path[i])
        img_hair.append(img)
    return img_hair

X=[]
y=[]

for i in range(len(hair_list)):
    X += get_img_hair(hair_list[i])
    y += [i]*len(get_img_hair(hair_list[i]))
X=np.array(X)
y=np.array(y)

rand_index = np.random.permutation(np.arange(len(X)))
X=X[rand_index]
y=y[rand_index]

X_train = X[:int(len(X)*0.8)]
y_tra





input_tensor=Input(shape=(64,64,3))
vgg16=VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

top_model=Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(len(hair_list), activation='softmax'))

model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))
for layer in model.layers[:15]:
    layer.trainsble = False
model.compile(loss='categorical_crossentropy', optomizer=optimizers.SGD(lr=1e-4, momentum=0.9))
model.save('model.h5')
