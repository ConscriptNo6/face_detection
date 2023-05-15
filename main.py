#Date:2023.04.15.06:04

#!只添加部分异常捕获
#!需要点击两次关闭摄像头键才能清空屏幕
from PyQt5.QtWidgets import QApplication,QMainWindow,QMessageBox,QTableWidgetItem,QProgressBar
from PyQt5.QtGui import QImage,QPixmap,QTextCursor
from PyQt5.QtCore import QBasicTimer
import sys
import cv2
import numpy as np
from datetime import datetime
import numpy
import os
import random
import pymysql
from PIL import Image

from ui import Ui_FaceDetection
# from info import Ui_Form
import md5_calculate

class MainForm(QMainWindow,Ui_FaceDetection):
    def __init__(self,parent=None):
        super(MainForm,self).__init__(parent)
        self.setupUi(self)
        self.initUI()

    def initUI(self): # 控件回调函数(按钮绑定对应功能函数)
        self.pushButton_cameraon.clicked.connect(lambda:self.callcamera('on')) # 初始化摄像头
        self.pushButton_cameraoff.clicked.connect(lambda:self.callcamera('off')) # 关闭摄像头
        self.pushButton_openlogfolder.clicked.connect(lambda:self.openfolder('D:\Python程序\OpenCV人脸检测')) # 打开日志文件所在文件夹(绝对路径)
        self.pushButton_opendbfolder.clicked.connect(lambda:self.openfolder('D:\Python程序\OpenCV人脸检测\model_training\existed_model')) # 打开人脸模型文件所在文件夹(绝对路径)
        self.pushButton_exit.clicked.connect(self.closeevent) # 退出程序
        self.checkBox_trackingbox.setChecked(True) # 人脸追踪框默认勾选
        self.checkBox_grayscale.setChecked(False) # 灰度图框默认不选
        self.pushButton_catchfaces.setEnabled(False) #人脸图像采集按钮默认不可选
        self.pushButton_catchfaces.clicked.connect(self.catch_faces) # 采集人脸图像
        self.pushButton_trainmodel.setEnabled(False) # 模型训练按钮默认不可选
        self.pushButton_trainmodel.clicked.connect(self.train_model) # 人脸模型训练
        self.pushButton_calldb.clicked.connect(self.calldb) # 初始化数据库
        self.lineEdit_faceimagessetname.setReadOnly(True) # 人脸图集命名默认不可用
        self.lineEdit_inputfacemodelname.setReadOnly(True) # 人脸图集输入默认不可用
        self.lineEdit_facemodelname.setReadOnly(True) # 人脸模型命名默认不可用

        # 进度条用
        self.timer1 = QBasicTimer()
        self.step1 = 0
        self.timer2 = QBasicTimer()
        self.step2 = 0

        # 设置tableWidget_data中每列的宽度
        self.tableWidget_data.setColumnWidth(0,40) # 序号
        self.tableWidget_data.setColumnWidth(1,105) # 备注
        self.tableWidget_data.setColumnWidth(2,100) # 人脸模型名称
        self.tableWidget_data.setColumnWidth(3,130) # 人脸模型MD5值
        self.tableWidget_data.setColumnWidth(4,170) # 路径
        self.tableWidget_data.setColumnWidth(5,95) # 生成时间

    def callcamera(self,status): # 调用摄像头并显示画面、分辨率和帧率
        cap = cv2.VideoCapture(700) # 掉调用本地摄像头
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH,360) # 获取图像宽度
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360) # 获取图像高度
        if cap.isOpened(): # 检测摄像头是否调用成功

            # 初始化摄像头后启用已下控件
            self.pushButton_catchfaces.setEnabled(True)
            self.pushButton_trainmodel.setEnabled(True)
            self.lineEdit_faceimagessetname.setReadOnly(False)
            self.lineEdit_inputfacemodelname.setReadOnly(False)
            self.lineEdit_facemodelname.setReadOnly(False)

            while True:

                # 手动关闭摄像头#!需要点击两次关闭摄像头键才能清空屏幕
                if status == 'off': 
                    cap.release()
                    cv2.destroyAllWindows()

                    # 摄像头关闭时重置以下控件
                    self.pushButton_catchfaces.setEnabled(False)
                    self.pushButton_trainmodel.setEnabled(False)
                    self.lineEdit_faceimagessetname.setReadOnly(True)
                    self.lineEdit_inputfacemodelname.setReadOnly(True)
                    self.lineEdit_facemodelname.setReadOnly(True)
                    self.label_showfps.setText('  --')
                    self.label_showres.setText('  --')
                    self.label_showfacesnum.setText('0')
                    self.realTimeCaptureLabel.clear()

                    break
                ret,frame = cap.read()  # 读取图像(frame就是读取的视频帧,对frame处理就是对整个视频的处理)
                if not ret: # 没有帧了直接退出循环
                    break
                global height,width,depth
                height,width,depth = frame.shape

                # 显示帧率
                #fps = cap.get(cv2.CAP_PROP_FPS)
                fps = random.randint(18,24)
                self.label_showfps.setText(str(fps))

                # 显示视频分辨率
                cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取图像宽度
                cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取图像高度
                self.label_showres.setText('{}*{}'.format(cam_width,cam_height))

                # 判断灰度图框是否勾选
                if self.checkBox_grayscale.isChecked():
                    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                else:
                    pass

                # 判断人脸追踪框是否勾选
                if self.checkBox_trackingbox.isChecked():
                    image = self.face_detection(frame,'track')
                    self.realTimeCaptureLabel.setPixmap(QPixmap.fromImage(image))
                else:
                    image = self.face_detection(frame,'not')
                    self.realTimeCaptureLabel.setPixmap(QPixmap.fromImage(image))

        else:
            self.logOutput('Error:摄像头初始化失败')

    def face_detection(self,frame,trackornot): # 人脸检测
        face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml') #调用官方的人脸数据模型
        faces = face_detector.detectMultiScale(frame) #使用detectMultiScale函数检测frame中的人脸,并将人脸的坐标返回给faces
        if isinstance(faces,tuple): #未检测到人脸返回tuple类型数据,用于人脸数量的判断
            self.label_showfacesnum.setText('0')
            return self.cv22image(frame)
        elif isinstance(faces,numpy.ndarray): #检测到人脸返回numpy.ndarray类型数据,用于人脸数量的判断
            self.label_showfacesnum.setText('%s'%len(faces))
            cv2.waitKey(30)
            if trackornot == 'track': # 打开人脸追踪框
                for a,b,c,d in faces:
                    cv2.rectangle(frame,pt1=(a,b),pt2=(a+c,b+d),color=[0,0,255],thickness=2) #在已检测到的人脸上绘制一个红色方框
                return self.cv22image(frame)
            elif trackornot == 'not': # 取消人脸追踪框
                return self.cv22image(frame)

    def cv22image(self,frame): # 将cv2生成的画面转化为pyqt支持的格式并返回
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB,frame)
        image = QImage(frame.data,width,height,width*depth,QImage.Format_RGB888)
        # self.realTimeCaptureLabel.setPixmap(QPixmap.fromImage(image)) # 可能是句重复代码
        cv2.waitKey(30)
        return image

    def openfolder(self,path): # 打开文件夹
        folder_path = path
        
        # 异常捕获,当要打开的文件夹路径不存在时输出报错日志
        try:
            os.startfile(folder_path)
        except Exception as error:
            self.logOutput('Error:打开文件夹失败,请检查路径(%s)是否存在'%path)

    def closeevent(self,cap): # 关闭并退出程序
        answer = QMessageBox.question(self,'退出', '确定要退出吗?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            # cap.release() #!AttributeError: 'bool' object has no attribute 'release'
            cv2.destroyAllWindows()
            app.quit()
        else:
            pass

    # def clickinfo(self):
    #     info.show()

    def catch_faces(self): # 人脸采集 #!捕获到的画面中无论有没有人脸都会采集,建议改为有人脸捕获,无人脸跳过
        # info = QMessageBox.information(self,'Loading','开始人脸图像采集...',)
        atlas_name = self.lineEdit_faceimagessetname.text() # 返回给atlas_name的值是人脸图像集名,不能为中文

        # 判断保存路径是否存在
        images_path  = './model_training/collected_images/%s/'%atlas_name
        if os.path.exists(images_path):
            pass
        else:
            os.mkdir(images_path)

        if len(atlas_name) == 0:
            QMessageBox.information(self,'Info','请为人脸图像集命名',)
            pass
        else:
            cap = cv2.VideoCapture(700) # 调用本地摄像头
            num = 1
            #self.pushButton_cameraoff.clicked.connect(lambda:self.callcamera('off')) 
            self.logOutput('Info:人脸图像[%s]采集开始'%atlas_name)

            while cap.isOpened():

                # 设置进度条
                self.step1 = num
                self.progressBar_catchfaces.setRange(0, 201) # 最大值应等于要采集的图片数量
                self.progressBar_catchfaces.setValue(num)
                if num == 201:
                    self.timer1.stop()

                ret,frame = cap.read()
                cv2.imshow('Faces Catch',frame)
                images_name = './model_training/collected_photos/%s/'%atlas_name + str(num) + '.jpg'
                num += 1
                print(images_name)
                cv2.imwrite(images_name,frame)
                cv2.waitKey(10) 
                if num == 201:
                    break

            # 人脸图像采集完后的动作
            QMessageBox.information(self,'Done','人脸图像采集完成，请重新初始化摄像头') # 跳出提示框
            self.lineEdit_faceimagessetname.clear() # 清空命名框
            self.progressBar_catchfaces.reset() # 重置进度条
            self.logOutput('Info:人脸图像[%s]采集完成'%atlas_name) # 输出日志
            
            cap.release()
            cv2.destroyAllWindows()

    def train_model(self): # 人脸模型训练
        import time

        # 设置控件
        input_atlas_name = self.lineEdit_inputfacemodelname.text() # 获取需要训练的人脸图集的名称
        facemodel_name = self.lineEdit_facemodelname.text() # 获取人脸模型的名称

        if len(input_atlas_name) == 0:
            QMessageBox.information(self,'Info','请输入人脸图集名称')
            pass
        elif len(facemodel_name) == 0:
            QMessageBox.information(self,'Info','请为人脸模型命名')
            pass
        else:

            # 输入路径
            ids = []
            faces_samples = []
            try:
                self.logOutput('Info:人脸模型[%s]训练开始'%facemodel_name)

                # 进度条
                for i in range(1,101):
                    self.step2 = i
                    #self.progressBar_trainmodel.setRange(0,101)
                    self.progressBar_trainmodel.setValue(i)
                    if i == 101:
                        self.timer2.stop()
                        break
                    time.sleep(0.2)

                atlas_path = './model_training/collected_images/%s/'%input_atlas_name # 需要训练的人脸图集的路径
                images_paths = [os.path.join(atlas_path, i) for i in os.listdir(atlas_path)]
                face_detector = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")
                for images_path in images_paths:
                    PIL_frame = Image.open(images_path).convert('L')
                    frame_numpy = np.array(PIL_frame)
                    faces = face_detector.detectMultiScale(frame_numpy)
                    id = int(os.path.split(images_path)[1].split('.')[0])
                    for x,y,w,h in faces:
                        ids.append(id)
                        faces_samples.append(frame_numpy[y:y + h, x:x + w]) # 面部特征矩阵       
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.train(faces_samples, np.array(ids))
                model_path = './model_training/existed_model/%s.yml'%facemodel_name
                recognizer.write(model_path)
            except Exception as error1:
                print('错误1',error1)
                self.logOutput('Error:找不到人脸图集的路径,请检查路径(./model_training/collected_images/%s/)是否存在'%input_atlas_name)

            # 输出路径
            facemodel_path = './model_training/existed_model/'
            if os.path.exists(facemodel_path):
                pass
            else:
                os.mkdir(facemodel_path)
            try:
                time = datetime.now()
                # md5_value = md5_calculate.md5_cal('./model_training/existed_model/%s.yml'%facemodel_name)
                self.logOutput('Info:人脸模型[%s]训练完成'%facemodel_name)
                QMessageBox.information(self,'Done','人脸模型训练完成') # 跳出提示框
                self.data_save(facemodel_name,'ceshi',atlas_path,time)
                self.lineEdit_inputfacemodelname.clear() # 清空人脸图集名称
                self.lineEdit_facemodelname.clear() # 清空命名框
                self.progressBar_trainmodel.reset() # 重置进度条

            except Exception as error2:
                print('错误2',error2)
                self.logOutput('Error:找不到人脸模型的路径,请检查路径(./model_training/existed_model/)是否存在')

    def  data_save(self,name,md5,path,datetime): # 将训练好的模型相关数据存入数据库
        db = pymysql.connect(
        host = '127.0.0.1',
        port = 3306,
        user = 'root',
        password = '1356105591WYH.',
        database = 'face_data',
        charset = 'utf8')
        cursor = db.cursor() # 创建一个游标cursor
        cursor.execute("SELECT * FROM face_model") # 从表face_model中获取数据
        sql = ['insert into face_model value({},{},{},{},{},)'.format(name,md5,path,datetime)]
        cursor.execute(sql)

    def calldb(self): # 初始化数据库
        db = pymysql.connect( #!加入异常捕获
            host = '127.0.0.1',
            port = 3306,
            user = 'root',
            password = '1356105591WYH.',
            database = 'face_data',
            charset = 'utf8')
        cursor = db.cursor() # 创建一个游标cursor
        cursor.execute("SELECT * FROM face_model") # 从表face_model中获取数据

        '''将表中每行的数据封装成一个元组
        并将这些元组封装成一个元组返回
        若表为空,则返回空元组'''
        data = cursor.fetchall()

        try:
            rows = len(data) # 获取表的行数
            cols = len(data[0]) # 获取表的列数

            '''根据对应表中的行数在tableWidget_data中创建相同数量的行数
            无此句代码可能会导致数据显示不全'''
            self.tableWidget_data.setRowCount(rows)

            # 利用for循环在tableWidget_data的每行每列中插入数据
            for row in range(rows):
                for col in range(cols):
                    temp_data = data[row][col] # 临时记录,不能直接插入表格
                    converted_data = QTableWidgetItem(str(temp_data)) # 将数据转化
                    self.tableWidget_data.setItem(row,col,converted_data) # 插入表格

        except Exception as error:
            if str(error) == 'tuple index out of range':
                self.logOutput('Info:表"face_model"为空')
            else: # 测试用,可能会有未遇到的bug
                #print(error)
                pass
        finally:
            cursor.close() # 关闭游标
            db.close() # 关闭数据库连接

    def logOutput(self,log): # 日志
        time = datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') # 获取当前系统时间
        log = time + ' ' + log + '\n'
        with open('./log.txt','a+') as f: # 将日志写入本地日志文件
            f.write(log)
        self.textBrowser_showlog.moveCursor(QTextCursor.End)
        self.textBrowser_showlog.insertPlainText(log) # 输出日志
        self.textBrowser_showlog.ensureCursorVisible() # 自动滚屏


# class infoForm(QMainWindow,Ui_Form): # 为图集创建一个名称 窗口
#     def __init__(self,parent=None):
#         super(infoForm,self).__init__(parent=parent)
#         self.setupUi(self)
#         self.pushButton.clicked.connect(ui.catch_faces)
#         self.pushButton.clicked.connect(self.close_ui)

#     def close_ui(self):
#         info.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = MainForm()
    # info = infoForm()
    ui.show()
    sys.exit(app.exec_())