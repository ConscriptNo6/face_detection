import cv2
import dlib

def eigenvalues_calculate(source):
    img = cv2.imread(source)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    detecter = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    faces = detecter(img_gray)
    for face in faces:
        x1,y1 = face.left(),face.top()
        x2,y2 = face.right(),face.bottom()
        cv2.rectangle(img_gray,(x1,y1),(x2,y2),(0,255,0),2)
        landmarks = predictor(img_gray,face)
        points = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append([x,y])
            cv2.circle(img_gray, (x, y), 5, (50,50,255),cv2.FILLED)
            cv2.putText(img_gray,str(n),(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,0,255),1)
    #     print(points)
    return points
    # cv2.imshow('test',img_gray)
    # cv2.waitKey(0)

if __name__ == '__name__':
    eigenvalues_calculate('./test.jpg')