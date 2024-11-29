# Create a Label Widget to display the text or Image
from tkinter import*
from tkinter import messagebox
from tkinter import Message, Text
import sqlite3
import cv2
import os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd 
import datetime
import time 
from pathlib import Path
import json
import random
import os
import string
import re
import dlib 
import imutils
from scipy.spatial import distance as dist
from imutils import face_utils


root=Tk()
root.title('LOCKER SYSTEM')
root.geometry('600x500+300+200')
root.configure(bg='blue')
root.resizable(False,False)



def signin():

    username=user.get()
    accountno1=accountno.get()

    # username='vidya'
    # accountno1=123

    if accountno1=='' or username=='':
        messagebox.showerror("empty","Enter the Registered Account_No and Username")

    # elif accountno1!='' or username!='':
    #     messagebox.showerror("Invalid","Invalid Credentials")

    else:
        conn = sqlite3.connect('ATM_System.db')
        cur = conn.execute('Select * from ATM where accountno="%s" AND user="%s"'%(accountno1,username))  

        if cur.fetchone():
            # frame.destroy()
            screen=Frame(root,width=400,height=400)
            screen.place(anchor='center', relx=0.5, rely=0.5)
            def generate_otp():
                global generateotp
                generateotp=random.randint(0000, 1000)
                print('Here is your generated OTP:', generateotp)
                return generateotp
            
            def verify_otp():
                    otp1=(otp.get())

                    print(otp1)
                    # print(generateotp)
                    if otp1=="":
                        messagebox.showerror("empty","Enter the geneated otp")
                        
                    # elif otp1!=generateotp:
                    #     messagebox.showerror("Wrong_OTP","Please Enter Valid OTP")
                    #     print("Invalid Otp")
                
                        

                    else:
                        otp1==generateotp
                        print("verification successfull")
                        messagebox.showinfo("verify","Verification Successfull")
                        print(otp1)
                        
                        def facedetection():
                            def draw_boundry(img,classifier,scaleFactor,minNeighours,color,text,clf):
                                gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                                features=classifier.detectMultiScale(gray_image,scaleFactor,minNeighours)

                                coord=[]

                                for (x,y,w,h) in features:
                                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
                                    id,predict=clf.predict(gray_image[y:y+h,x:x+w])
                                    confidence=int((100*(1-predict/300)))

                                    conn=sqlite3.connect('ATM_System.db')
                                    with conn:
                                        cur=conn.cursor()
                                        cur.execute("select user from ATM where atmpin="+str(id))
                                        un=cur.fetchone()
                                        un = (un[0])
                                        print(un)

                                        cur.execute("select accountno from ATM where atmpin="+str(id))
                                        a=cur.fetchone()
                                        a = (a[0])   
                                        print(a)

                                        if confidence>77:
                                            cv2.putText(img,f"UserName:{un}",(x,y-55),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)

                                            cv2.putText(img,f"AccountNo:{a}",(x,y-30),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)

                                            screen1=Frame(root,width=400,height=400)
                                            screen1.place(anchor='center', relx=0.5, rely=0.5)

                                            def blink_detection():
                                                print("hello")
                                                cam = cv2.VideoCapture(0)

                                                # defining a function to calculate the EAR
                                                def calculate_EAR(eye):

                                                    # calculate the vertical distances
                                                    y1 = dist.euclidean(eye[1], eye[5])
                                                    y2 = dist.euclidean(eye[2], eye[4])

                                                    # calculate the horizontal distance
                                                    x1 = dist.euclidean(eye[0], eye[3])

                                                    # calculate the EAR
                                                    EAR = (y1+y2) / x1
                                                    return EAR

                                                # Variables
                                                blink_thresh = 0.45
                                                succ_frame = 2
                                                count_frame = 0

                                                # Eye landmarks
                                                (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
                                                (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

                                                # Initializing the Models for Landmark and
                                                # face Detection
                                                detector = dlib.get_frontal_face_detector()
                                                landmark_predict = dlib.shape_predictor(
                                                    'shape_predictor_68_face_landmarks.dat')
                                                while 1:
                                                        _, frame = cam.read()
                                                        frame = imutils.resize(frame, width=640)

                                                        # converting frame to gray scale to
                                                        # pass to detector
                                                        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                                                        # detecting the faces
                                                        faces = detector(img_gray)
                                                        for face in faces:

                                                            # landmark detection
                                                            shape = landmark_predict(img_gray, face)

                                                            # converting the shape class directly
                                                            # to a list of (x,y) coordinates
                                                            shape = face_utils.shape_to_np(shape)

                                                            # parsing the landmarks list to extract
                                                            # lefteye and righteye landmarks--#
                                                            lefteye = shape[L_start: L_end]
                                                            righteye = shape[R_start:R_end]

                                                            # Calculate the EAR
                                                            left_EAR = calculate_EAR(lefteye)
                                                            right_EAR = calculate_EAR(righteye)

                                                            # Avg of left and right eye EAR
                                                            avg = (left_EAR+right_EAR)/2
                                                            if avg < blink_thresh:
                                                                count_frame += 1 # incrementing the frame count
                                                            else:
                                                                if count_frame >= succ_frame:
                                                                    cv2.putText(frame, 'Blink Detected', (30, 30),
                                                                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)

                                                                    screen2=Frame(root,width=400,height=400)
                                                                    screen2.place(anchor='center', relx=0.5, rely=0.5)

                                                                    Label(screen2,text=f"Welcome:{un} in Locker System",font=('Microsoft YaHei UI Light',11,'bold')).place(x=20,y=20)

                                                                    Button(screen2,width=20,pady=7,text='Gold_Locker',bg='#57a1f8',fg='white',border=2,font=('Microsoft YaHei UI Light',11,'bold')).place(x=100,y=80)
                                                                    
                                                                    Button(screen2,width=20,pady=7,text='Silver_Locker',bg='#57a1f8',fg='white',border=2,font=('Microsoft YaHei UI Light',11,'bold')).place(x=100,y=170)

                                                                    Button(screen2,width=20,pady=7,text='Valuable_Things',bg='#57a1f8',fg='white',border=2,font=('Microsoft YaHei UI Light',11,'bold')).place(x=100,y=250)


                                                                else:
                                                                    count_frame = 0

                                                        cv2.imshow("Video", frame)
                                                        if cv2.waitKey(5) & 0xFF == ord('q'):
                                                            break

                                                cam.release()
                                                cv2.destroyAllWindows()



                                            Label(screen1,text="VERIFY WITH EYE BLINK DETECTION",font=('Microsoft YaHei UI Light',11,'bold')).place(x=30,y=40)

                                            Button(screen1,width=20,pady=7,text='Eye_Blink_Detection',bg='#57a1f8',fg='white',border=2,font=('Microsoft YaHei UI Light',11,'bold'),command=blink_detection).place(x=100,y=100)
                                            

                                           

                                        else:
                                            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
                                            cv2.putText(img,"Uknown Face",(x,y-55),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)

                                        coord=[x,y,w,h]
                                return coord  

                            def recognize(img,clf,faceCascade):
                                coord=draw_boundry(img,faceCascade,1.1,10,(255,25,255),"Face",clf)
                                return img
                            
                            faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
                            clf=cv2.face.LBPHFaceRecognizer_create()
                            clf.read("classifier.xml")

                            cap=cv2.VideoCapture(0)

                            while True:
                                ret,img=cap.read()
                                img=recognize(img,clf,faceCascade)
                                cv2.imshow("Welcome to face recognizer",img)

                                if cv2.waitKey(1)==13:
                                    break
                            cap.release()
                            cv2.destroyAllWindows()

                                  

                            
                        Button(screen,width=20,pady=7,text='FaceDetection',bg='#57a1f8',fg='white',border=2,font=('Microsoft YaHei UI Light',11,'bold'),command=facedetection).place(x=100,y=310)
                
            def back():
                screen.destroy()
            
            

            Button(screen,width=20,pady=7,text='GENERATE OTP',bg='#57a1f8',fg='white',border=2,font=('Microsoft YaHei UI Light',11,'bold'),command=generate_otp).place(x=100,y=80)

            Label(screen,text='ENTER OTP:').place(x=40,y=180)
            otp = Entry(screen,width=25,fg='black',font=('Microsoft YaHei UI Light',11,'bold'))
            otp.place(x=110,y=180)

            Button(screen,width=20,pady=7,text='VERIFY OTP',bg='#57a1f8',fg='white',border=2,font=('Microsoft YaHei UI Light',11,'bold'),command=verify_otp).place(x=100,y=250)
            
            global back1
            back1 = Button(screen,width=20,pady=7,text='Back',bg='#57a1f8',fg='white',border=2,font=('Microsoft YaHei UI Light',11,'bold'),command=back).place(x=100,y=310)
            

#==============================================    
def signup():
    
    def signup1():
        username=user.get()
        accountno1=accountno.get()
        atmpin1=atmpin.get()
        mobileno1=mobileno.get()
        email_id=email.get()
        address=address1.get()

        if username=='' or accountno1=='' or atmpin1=='' or mobileno1=='' or email_id =='' or address=='':
            messagebox.showerror("empty","empty fields are not allowed")

        elif not mobileno1.isdigit() or len(mobileno1) != 10:
            messagebox.showerror("invalid","Invalid Mobile_No")

        elif len(accountno1) !=6:
            messagebox.showerror("invalid","Invalid Account_No")
            
        elif not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',email_id):
            messagebox.showerror("invalid","Invalid email_id")
        
        else:
            conn=sqlite3.connect('ATM_System.db')
            with conn:
               cur=conn.cursor()
               cur.execute('CREATE TABLE IF NOT EXISTS ATM (user TEXT, accountno INT, atmpin INT, mobileno INT, email INT, address1 TEXT)')
               

            cur.execute('INSERT INTO ATM (user,accountno,atmpin,mobileno,email,address1) VALUES(?,?,?,?,?,?)',(username,accountno1,atmpin1,mobileno1,email_id,address))
            conn.commit()
            messagebox.showinfo("success","Registration successfull")

            def generate_dataset():
                print("welcome")
                conn=sqlite3.connect('ATM_System.db')
                with conn:
                    cur=conn.cursor()
                    cur.execute("select * from ATM")
                    myrecord=cur.fetchall()
                    id=0
                    for x in myrecord:
                        id+=1
                conn.commit()
                conn.close()

                face_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

                def face_crop(img):
                    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    faces=face_classifier.detectMultiScale(gray,1.3,5)

                    for (x,y,w,h) in faces:
                        face_crop=img[y:y+h,x:x+w]
                        return face_crop
                    

                cap=cv2.VideoCapture(0)
                img_id=0
                while True:
                    ret,my_frame=cap.read()
                    face = face_crop(my_frame)
                    if face is not None:
                        img_id+=1
                        face=cv2.resize(face,(450,450))
                        face=cv2.cvtColor(face,cv2.COLOR_BGR2BGRA)
                        dataset_path="dataset/user."+str(id)+"."+str(img_id)+".jpg"
                        cv2.imwrite(dataset_path,face)
                        cv2.putText(face,str(img_id),(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
                        cv2.imshow("croped face",face)

                    if cv2.waitKey(1)==13 or int(img_id)==30:
                        break
                cap.release()
                cv2.destroyAllWindows()

                messagebox.showinfo("result","generating dataset completed")

                def train_dataset():
                    data_dir=("dataset")
                    path=[os.path.join(data_dir,file) for file in os.listdir(data_dir)]

                    faces=[]
                    ids=[]

                    for image in path:
                        img=Image.open(image).convert('L')
                        imageNP=np.array(img,'uint8')
                        id=int(os.path.split(image)[1].split('.')[1])

                        faces.append(imageNP)
                        ids.append(id)
                        cv2.imshow("Training",imageNP)
                        cv2.waitKey(1)==13
                    ids=np.array(ids)

                    clf=cv2.face.LBPHFaceRecognizer_create()
                    clf.train(faces,ids)
                    clf.write("classifier.xml")
                    cv2.destroyAllWindows()
                    messagebox.showinfo("tran","Training dataset completed")
                train_dataset()
                frame.destroy()
            generate_dataset()     

            
                  
                        
     # =================REGISTRATION FRAME========================    

    frame=Frame(root,width=400,height=400)
    frame.place(anchor='center', relx=0.5, rely=0.5)

    heading=Label(frame,text='REGISTRATION',fg='#57a1f8',font=('Microsoft YaHei UI Light',25,'bold'))
    heading.place(x=100,y=10)
     
    label = Label(frame,text='Username:',font=("bold")).place(x=40,y=80)
    user = Entry(frame,width=25,fg='black',border=1,bg="white",font=('Microsoft YaHei UI Light',11))
    user.place(x=140,y=80)
    
    label = Label(frame,text='Account_No:',font=("bold")).place(x=40,y=160)
    accountno = Entry(frame,width=25,fg='black',border=1,bg="white",font=('Microsoft YaHei UI Light',11))
    accountno.place(x=140,y=160)
    
    label = Label(frame,text='User_Id:',font=("bold")).place(x=40,y=120)
    atmpin = Entry(frame,width=25,fg='black',border=1,bg="white",font=('Microsoft YaHei UI Light',11))
    atmpin.place(x=140,y=120)

    label = Label(frame,text='Mobile_No:',font=("bold")).place(x=40,y=200)
    mobileno = Entry(frame,width=25,fg='black',border=1,bg="white",font=('Microsoft YaHei UI Light',11))
    mobileno.place(x=140,y=200)
    
    label = Label(frame,text='Email_Id:',font=("bold")).place(x=40,y=240)
    email = Entry(frame,width=25,fg='black',border=1,bg="white",font=('Microsoft YaHei UI Light',11))
    email.place(x=140,y=240)
   
    label = Label(frame,text='Address:',font=("bold")).place(x=40,y=280)
    address1 = Entry(frame,width=25,fg='black',border=1,bg="white",font=('Microsoft YaHei UI Light',11))
    address1.place(x=140,y=280)
     
    Button(frame,width=20,pady=7,text='Sign Up',bg='#57a1f8',fg='white',border=1,font=("bold"),command=signup1).place(x=100,y=330)
    

#==================================LOGIN_FRAME==============================

# bg =PhotoImage("atm\atm_login.jpg")
frame=Frame(root,width=400,height=400)
frame.place(anchor='center', relx=0.5, rely=0.5)

# Create an object of tkinter ImageTk
# img = ImageTk.PhotoImage(Image.open("atm\\l222.png"))

# Create a Label Widget to display the text or Image
# label = Label(frame,width=100,height=100, image = img)
# label.place(x=150,y=90)

heading=Label(frame,text='LOGIN',fg='#57a1f8',font=('Microsoft YaHei UI Light',40,'bold'))
heading.place(x=120,y=15)

label = Label(frame,text='Username:',font=("bold")).place(x=40,y=100)
user = Entry(frame,width=25,fg='black',border=1,bg="white",font=('Microsoft YaHei UI Light',11))
user.place(x=140,y=100)

label = Label(frame,text='Account_No:',font=("bold")).place(x=40,y=140)
accountno = Entry(frame,width=25,fg='black',border=1,bg="white",font=('Microsoft YaHei UI Light',11))
accountno.place(x=140,y=140)

Button(frame,width=20,pady=7,text='Sign in',bg='#57a1f8',fg='white',border=2,font=("bold"),command=signin).place(x=110,y=200)


# Button(frame,width=20,pady=7,text='FaceDetection',bg='#57a1f8',fg='white',border=2,font=("bold"),command=signin).place(x=110,y=200)

label=Label(frame,text="Don't have an account",fg='black',bg='white',font=('Microsoft YaHei UI Light',9,'bold'))
label.place(x=120,y=330)

sign_up = Button(frame,width=20,pady=7,text='Sign up',bg='#57a1f8',fg='white',border=2,cursor='hand2',font=("bold"),command=signup).place(x=110,y=275)



root.mainloop()

       
