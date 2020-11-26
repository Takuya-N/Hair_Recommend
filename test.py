import json
import requests
import sys
from funcs import fail_login,index,suc_login,logout,register,delete
#if sys.argv[1]=='destroy':
    #url = 'http://127.0.0.1:5000/destroy'
    #body = {"username":"x" , 'password':'x'}
    #r = requests.post(url,data=body)

fail_login()
index(code=401)
suc_login()
index(code=200)
logout()
index(code=401)
register()
delete()
