import requests
from bs4 import BeautifulSoup
import math

#URLをリスト化
url_list = ['https://beauty.hotpepper.jp/CSP/bt/hairCatalogSearch/mens/condtion/?lengthCd=HL09&pn=',
          'https://beauty.hotpepper.jp/CSP/bt/hairCatalogSearch/mens/condtion/?lengthCd=HL13&pn=',
          'https://beauty.hotpepper.jp/CSP/bt/hairCatalogSearch/mens/condtion/?keyword=%E3%83%9E%E3%83%83%E3%82%B7%E3%83%A5&pn=',
          'https://beauty.hotpepper.jp/CSP/bt/hairCatalogSearch/mens/condtion/?keyword=%E3%83%91%E3%83%BC%E3%83%9E&pn=',
          'https://beauty.hotpepper.jp/CSP/bt/hairCatalogSearch/mens/condtion/?keyword=%E3%83%84%E3%83%BC%E3%83%96%E3%83%AD%E3%83%83%E3%82%AF&pn=']
path_list = ['bouzu/bouzu','long/long','mash/mash','pama/pama','two_block/two_block']
#URLのデータをfile_nameに保存する関数
def download_img(url, file_name):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(file_name, 'wb') as f:
            f.write(r.content)
#50ページ分のURLから画像を取得
for j in range(len(url_list)):
    for i in range(50):
        url = url_list[j]+str(i+1)
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, 'lxml')
        img_list = soup.find_all('img', class_='imgFrame')
        count = 1
        for img in img_list:
            img_url = img['src']
            download_img(img_url, path_list[j]+str(i+1)+'-'+str(count)+'.png')
            count += 1