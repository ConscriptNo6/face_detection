# coding=utf-8
# 2023.02.02.00:21
# 计算出识别到的人脸的特征值的哈希值

import hashlib

def hash_calculate(eigenvalues): # 参数为识别到的人脸的特征值
    hash = hashlib.md5() # 创建一个md5算法的对象
    hash.update(bytes(eigenvalues,encoding='utf-8')) # 传入需要加密的字符串
    md5_value = hash.hexdigest() # 生成md5值
    return md5_value # 返回计算出的md5值

if __name__ == '__main__':
    hash_calculate()