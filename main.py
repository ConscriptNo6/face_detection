#coding=utf-8
#90-2023.01.25.15:25

import cv2
import numpy
from PIL import ImageFont, ImageDraw, Image
from alive_progress import alive_bar
import time

# def zh_char_output(input_source,char,char_size,x,y): #使用pillow库以在OpenCV窗口中输出中文
#     font = ImageFont.truetype('SmileySans-Oblique.ttf',char_size) #导入字体文件
#     source_pil = Image.fromarray(input_source) #创建一个pillow图片
#     draw = ImageDraw.Draw(source_pil)
#     draw.text((x,y),char,font=font) #利用draw去绘制中文
#     output_source = numpy.array(source_pil) #重新变成ndarray
#     return output_source

def motion_detected(input_source,width,height): #!当视频/图片中出现两张及以上人脸时,只会在其中一个检测框下显示'检测到人脸'
    cap = cv2.VideoCapture(input_source)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(frames)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)  #设置图像宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)  #设置图像高度
    while True:
        ret,frame = cap.read()  # 读取图像(frame就是读取的视频帧，对frame处理就是对整个视频的处理)早睡早起身体好（——  ——）
        if not ret: #没有帧了(即视频播放完毕)直接退出循环
            break
        source = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  # 将图像灰度化处理
        face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml') #调用官方的人脸数据模型
        faces = face_detector.detectMultiScale(source) #使用detectMultiScale函数检测source中的人脸,并将人脸的坐标返回给faces
        if isinstance(faces,tuple): #未检测到人脸返回tuple类型数据
            print('\033[1;31;40m未检测到人脸\033[0m')
            cv2.imshow('Motion Detection',source)
            cv2.waitKey(20)
        elif isinstance(faces,numpy.ndarray): #检测到人脸返回numpy.ndarray类型数据
            print('\033[1;34m已检测到%s张人脸\033[0m'%len(faces))
            source_pil = Image.fromarray(source) #创建一个pillow图片
            draw = ImageDraw.Draw(source_pil)
            for a,b,c,d in faces:
                #cv2.putText(source, 'face detected', (a,b), cv2.FONT_ITALIC, 1, (0,255,255), 4) #在方框上方绘制文字"face detected",只能绘制英文字符,不能绘制中文字符
                font = ImageFont.truetype('SmileySans-Oblique.ttf',30) #导入字体文件
                draw.text((a,b+c),'检测到人脸',font=font) #利用draw去绘制中文
                output_source = numpy.array(source_pil) #重新变成ndarray
                cv2.rectangle(output_source,pt1=(a,b),pt2=(a+c,b+d),color=[0,0,255],thickness=2) #在已检测到的人脸上绘制一个红色方框
                cv2.imshow('Motion Detection',output_source) #!cv2.imshow()不能出现在for循环内,否则只会在其中一个检测框下显示'检测到人脸'
                input_str = cv2.waitKey(20)
        # if input_str == ord('q'): #如果输入的是q就break,结束图像显示,鼠标点击视频画面输入字符 调试用
        #     break
    cap.release()  # 释放视频对象
    cv2.destroyAllWindows()

def camera(): #摄像头实时检测
    camera_index = 700
    width = 540
    height = 480
    motion_detected(camera_index,width,height)

def local_vedio(): #本地视频检测
    vedio_path = './test.mp4'
    motion_detected(vedio_path,None,None)

# def image_static_detection(): #本地图片检测
#     img = cv2.imread('./test.jpg')
#     face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml') #调用官方的人脸数据模型
#     faces = face_detector.detectMultiScale(img) #使用detectMultiScale函数检测img中的人脸,并将人脸的坐标返回给faces
#     for a,b,c,d in faces:
#         cv2.rectangle(img,pt1=(a,b),pt2=(a+c,b+d),color=[0,0,255],thickness=2) #在已检测到的人脸上绘制一个红色方框
#     cv2.imshow('static detection',zh_char_output(img,'检测到人脸',30,a,b+c))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

if __name__ == '__main__':
    t1 = time.time()
    local_vedio()
    #camera()
    #image_static_detection()
    t2 = time.time()
    t0 = t2 - t1
    print('运行时间为%s:'%t0)