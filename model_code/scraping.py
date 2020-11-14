import requests
from bs4 import BeautifulSoup
import math


base_url ='https://beauty.hotpepper.jp/CSP/bt/hairCatalogSearch/mens/condtion/?keyword=%E3%83%84%E3%83%BC%E3%83%96%E3%83%AD&isSearchName=true&pn='


def download_img(url, file_name):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(file_name, 'wb') as f:
            f.write(r.content)
    

for i in range(50):
    url = base_url+str(i+1)
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'lxml')
    img_list = soup.find_all('img', class_='imgFrame')
    count = 1
    for img in img_list:
        img_url = img['src']
        download_img(img_url, 'two_block/two_block'+str(i+1)+'-'+str(count)+'.png')
        count +=1