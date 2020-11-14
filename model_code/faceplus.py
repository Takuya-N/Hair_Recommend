import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

path_bouzu =[]
path_long = []
path_mash = []
path_pama = []
path_two_block = []


hair_list=['bouzu','long','mash','pama','two_block']
path_list=[path_bouzu,path_long,path_mash,path_pama,path_two_block]




def scratch_image(img):
    img_size = img.shape
    filter1 = np.ones((3,3))
    images = [img]
    scratch = np.array([lambda x: cv2.flip(x,1), 
                        lambda x: cv2.threshold(x, 150,255,cv2.THRESH_TOZERO)[1],
                        lambda x: cv2.GaussianBlur(x,(5,5),0),
                        lambda x: cv2.erode(x,filter1)])
    doubling_images = lambda f, imag: (imag + [f(i) for i in imag])
    for func in scratch:
        images = doubling_images(func, images)
    return images  

for i in range(len(hair_list)):
    path = './'+hair_list[i]+'_face'
    path_list[i] = glob.glob(path+'/*')
    print(path_list[i])
    for img_path in path_list[i]:
        
        img_path = img_path.replace('\\','/')
        base_name = os.path.basename(img_path)
        name, ext = os.path.splitext(base_name)
        img = cv2.imread(img_path)
        print(img.shape)
        
        scratch_images = scratch_image(img)
        for num, im in enumerate(scratch_images):
            cv2.imwrite('./'+hair_list[i]+'_face/'+name+str(num+100)+ext,im)
    






