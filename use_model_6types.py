import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image
import time,datetime

#import graph

#face detection model
cascade_path="models/haarcascade_frontalface_default.xml"
cascade=cv2.CascadeClassifier(cascade_path)
image_path='temp_train/temp.jpg'

#read the traind model
model_path = 'models/6face_emotions_300ep.hdf5'
classes = ({0:'angry',1:'disgust',2:'happy',3:'neutral',4:'sad',5:'surprise'})
emotions_XCEPTION = load_model(model_path, compile=False)

#print result information
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontsize = 0.8; fontwidth = 2
colors = {'blue':(255,0,0),'coral':(80,127,255),'light coral':(128,128,240)}# these are BGR. typically RGB is common.
pertime=100# 1/1000sencond

#web camera on
capture = cv2.VideoCapture(0)

timeemos=[]
#print("----------------------- start -----------------------")
#startTime=time.time()
A=0
while(True):
        # read web camera
        ret,frame=capture.read()

        # execute the face detection model
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # transform the picture to a gray one
        front_face_list=cascade.detectMultiScale(gray,minSize=(50,50))
        #print(front_face_list)
        if A==0:
            startTime2=time.time()

        # In case of detecting no face
        if len(front_face_list) ==0:
            cv2.putText(frame, "No Face", (10,30),fontface,fontsize,colors['blue'],fontwidth)
            print("Failed")
            cv2.imshow("frame_orig",frame)
            key = cv2.waitKey(pertime)
            timeemos.append([0,0,0,0,0,0,0])
            A=A+1
            continue

        # In case of detecting faces
        else:
            for (x,y,w,h) in front_face_list:
                #save pictures of detected faces as jpg files
                img1 = frame[y : y+h, x: x+w]
                cv2.imwrite(image_path, img1)

                # execute the traind model
                img = image.load_img(image_path, grayscale=True , target_size=(48, 48))
#                img = image.load_img(image_path, grayscale=False , target_size=(48, 48))
                img_array = image.img_to_array(img)
                pImg = np.expand_dims(img_array, axis=0) / 255
                prediction = emotions_XCEPTION.predict(pImg)[0]
                #print(prediction)

                # make the image with the result of the traind model
                for i in range(len(prediction)):
                    cv2.putText(frame, (str(classes[i]) + '=' + str(int(prediction[i]*100))+'%'), (x+w+5,y+20+i*30),fontface,fontsize,colors['coral'],fontwidth)

                # make the image of the result of the face detection
                cv2.rectangle(frame,(x,y),(x+w,y+h),colors['light coral'],thickness=2)

                # output the screen
                cv2.imshow("frame_orig",frame)
                key = cv2.waitKey(pertime)

                emos=[]
                for i in range(len(prediction)):
                    emos.append(prediction[i])
                timeemos.append(emos)
                A=A+1
                continue

        if key != -1:
            cv2.destroyAllWindows()
            break
endTime=time.time()

# show the result
#print("time of Program processing is "+str(endTime-startTime))
print("time of Program processing is "+str(endTime-startTime2))
print(str(A)+"times for distinguishing")
print(timeemos)

dt = datetime.datetime.now()
now = dt.strftime('%Y-%m-%d_%H_%M_%S')
f = open("emomemo/"+now+".txt", "w")

# write the data to made textfile
for i in range(len(timeemos)):
    for j in range(len(timeemos[i])):
        f.write(str(timeemos[i][j]))
        if j != len(timeemos[i])-1:
            f.write(",")
    if i != len(timeemos)-1:
        f.write("\n")
f.close()
