import json
import requests
import sys
session = requests.Session()

#ログイン失敗の場合
def fail_login():
    global session
    url = 'http://127.0.0.1:5000/'
    body = {"username":"aaa" , 'password':'aaa'}
    try:
        r = session.post(url,data=body)
        if r.status_code == 401:
            print('ログイン失敗の場合...正常処理')
        else:
            print('ログイン失敗の場合...異常処理')
            print(r)
    except:
        print('ログイン失敗の場合に予期せぬエラーが発生しました')

#ログイン成功の場合
def suc_login():
    global session
    url = 'http://127.0.0.1:5000/'
    body = {"username":"d" , 'password':'d'}
    try:
        r = session.post(url,data=body)
        if r.status_code == 200:
            print('ログイン成功の場合...正常処理')
        else:
            print('ログイン成功の場合...異常処理')
            print(r)
    except:
        print('ログイン成功の場合に予期せぬエラーが発生しました')

#indexの確認
#コード=200が正常、401が異常
def index(code):
    global session
    url = 'http://127.0.0.1:5000/index'
    try:
        r = session.get(url)
        if r.status_code == code:
            print('indexの確認の場合...正常処理')
        else:
            print('indexの確認の場合...異常処理')
            print(r)
    except:
        print('indexの確認の場合に予期せぬエラーが発生しました')

#登録の場合
def register():
    url = 'http://127.0.0.1:5000/register'
    body = {"username":"x6" , 'password':'x6'}
    try:
        r = requests.post(url,data=body)
        if r.status_code == 200:
            print('登録の場合...正常処理')
        else:
            print('登録の場合...異常処理')
            print(r)
    except:
        print('登録の場合に予期せぬエラーが発生しました')

#削除の場合
def delete():
    url = 'http://127.0.0.1:5000/destroy'
    body = {"username":"x6" , 'password':'x6'}
    try:
        r = requests.post(url,data=body)
        if r.status_code == 200:
            print('削除の場合...正常処理')
        else:
            print('削除の場合...異常処理')
            print(r)
    except:
        print('削除の場合に予期せぬエラーが発生しました')

#ログアウトの場合
def logout():
    global session
    url = 'http://127.0.0.1:5000/logout'
    body = {"username":"aaa" , 'password':'aaa'}
    try:
        r = session.get(url)
        if r.status_code == 200:
            print('ログアウトの場合...正常処理')
        else:
            print('ログアウトの場合...異常処理')
            print(r)
    except:
        print('ログアウトの場合に予期せぬエラーが発生しました')
