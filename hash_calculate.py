# coding=utf-8
# 2023.04.03.05:01
# 计算出人脸模型的md5值

import hashlib

def md5_cal(file_path):
    md5 = hashlib.md5()
    with open(file_path, 'rb') as file:
        while True:
            data = file.read(4096)   #定义每次读只读4096字节
            if not data:                  #如果data没有数据了则break
                break
            md5.update(data)         #更新数据
    return md5.hexdigest()
