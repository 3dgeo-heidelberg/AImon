import os
import requests
from getpass import getpass
import json




def upload(folder_pc, login_url, upload_url):
    client = requests.session()
    # Retrieve the CSRF token first
    client.get(login_url)  # sets cookie
    #print(client.cookies)
    csrftoken = client.cookies['sfcsrftoken']
    
    id = input("Enter your id: ")
    pswd = getpass("Enter your password: ", )
    login_data = dict(username=id, password=pswd, csrfmiddlewaretoken=csrftoken, next='/')

    r = client.post(login_url, data=login_data, headers=dict(Referer=login_url))

    listdir_pc = os.listdir(folder_pc)
    for file_pc in listdir_pc:
        path_file = f"{folder_pc}{file_pc}"
        print(path_file)
        with open(path_file, 'rb') as f:
            r = client.post(upload_url, data=login_data, files = {'upload_pc': f})
            r = client.post(upload_url, files = {'upload_pc': f})

        if r.status_code == 200:
            print("Upload completed successfully!")
            print(r.text)
        else:
            print("Upload failed with status %s" % r.status_code)
        
        break

    pass

def upload_GDrive(folder_pc, url):
    listdir_pc = os.listdir(folder_pc)
    # The header is important is you need a token
    # For GDrive servers, we need a token that can be generated following these steps: https://www.delftstack.com/howto/python/python-upload-file-to-google-drive/
    headers = {"Authorization": "Bearer ya29.a0AcM612yLJA_jmOBFZB-MuP84MKBuqBWUOyJLZKF8jb9Qs0sVTPvoGvgOtDmYBhYcrxvC9Ub8oT20A7tepwnm2UybHv-R1T4uCwgzH4PymAfwcTS-dCRpVVxTYu6jSjiq2cqgMAphhoNBZAGjOJAShvSDCQQsvilihJS-WiZ5aCgYKAdgSARASFQHGX2MiFEv8KSwXZ75rWfB7R5P01g0175"}
    file_path = f"{folder_pc}{listdir_pc[0]}"

    para = {
        "name": f"{listdir_pc[0]}"
    }
    file_pc = {
        "data": ("metadata", json.dumps(para), "application/json; charset=UTF-8"),
        "file": open(file_path, "rb"),
    }

    # Transfer laz file
    r = requests.post(
        url,
        headers=headers,
        files=file_pc
    )

    
    file_json = {
        "data": "jsonfile",
        "key": "value"
    }
    # Transfer json file
    r = requests.post(
        url,
        headers=headers,
        json=file_json,
        name="jsonfile.json"
    )
    print(r.text)


if __name__ == '__main__':
    folder_pc = "/home/william/Documents/Work/4wilhelmVonRonno/pls_out/"
    login_url = "https://heibox.uni-heidelberg.de/accounts/login/?next=/"
    upload_url = "https://heibox.uni-heidelberg.de/u/d/1c88e75875e0410f97aa/"
    upload_url_GDrive = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
    upload_GDrive(folder_pc, upload_url_GDrive)
    #upload(folder_pc, login_url, upload_url)


