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
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector


root=Tk()
root.title('Bank_Locker_SYSTEM')
root.geometry('600x500+300+200')
root.configure(bg='blue')
root.resizable(False,False)



def signin():
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('TrainingImageLabel/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX

    detector = FaceMeshDetector(maxFaces=1)
    idList = [22,23,24,26,110,157,158,159,160,161,130,243]
    #iniciate id counter
    id = 0
    # names related to ids: example ==> viya: id=1,  etc
    names = ['None', 'vidya'] 
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height
    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    while True:
        
        ret, img =cam.read()
        # print(img)
        img,faces=detector.findFaceMesh(img,draw=False)
        if faces:
            face = faces[0]
            for id in idList:
                cv2.circle(img,face[id],5,(255,0,255),cv2.FILLED)
            
            leftup = face[159]
            leftdown = face[23]
            lengthhor,_ = detector.findDistance(leftup,leftdown)
            cv2.line(img,leftup,leftdown,(0,200,0),2)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            
            # If confidence is less them 100 ==> "0" : perfect match 
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
               
                screen=Toplevel(root)
                screen.title("OTP VERIFICATION")
                screen.geometry('600x500+300+200')
                screen.config(bg='blue')
                                
                def user_details():
                    
                    username=user.get()
                    atmpin1=atmpin.get()

                    if atmpin1=='' or username=='':
                        messagebox.showerror("empty","Enter the Registered Locker Pin and Username")

                    else:
                        conn = sqlite3.connect('ATM_System.db')
                        cur = conn.execute('Select * from ATM where atmpin="%s" AND user="%s"'%(atmpin1,username))  

                        if cur.fetchone():
                            print("hello")
                            global generateotp
                            generateotp=random.randint(0000, 1000)
                            print('Here is your generated OTP:', generateotp)
                            return generateotp

                def verify_otp():
                    print("verify")
                    otp1=int(otp.get())
                    # print(otp)
                    # print(generateotp)
                    if otp1==generateotp:
                        print("verification successfull")
                        messagebox.showinfo("verify","Verification Successfull")
                        screen.destroy()
                        screen1=Toplevel(root)
                        screen1.title("Locker SYSTEM")
                        screen1.geometry('600x500+300+200')
                        screen1.config(bg='blue')

                        # conn=sqlite3.connect('ATM_System.db')
                        # with conn:
                        #     cur=conn.cursor()
                        #     cur.execute('CREATE TABLE IF NOT EXISTS balance (withdraw TEXT, accountno INT, atmpin INT, mobileno INT, email INT, address1 TEXT)')
               

                        #     conn.commit()

                        
                        def cash_withdrawl():
                             screen1.destroy()
                             screen2=Toplevel(root)
                             screen2.title("Locker_1")
                             screen2.geometry('600x500+300+200')
                             screen2.config(bg='blue')

                             frame2=Frame(screen2,width=300,height=300)
                             frame2.place(anchor='center', relx=0.5, rely=0.5)

                             Label(frame2,text='Confidential_Data',font=('Microsoft YaHei UI Light',14,'bold')).place(x=60,y=120)

                         
                        def cash_deposite():
                             screen1.destroy()
                             screen3=Toplevel(root)
                             screen3.title("Locker_2")
                             screen3.geometry('600x500+300+200')
                             screen3.config(bg='blue')

                             frame2=Frame(screen3,width=300,height=300)
                             frame2.place(anchor='center', relx=0.5, rely=0.5)

                             Label(frame2,text='Confidential_Data',font=('Microsoft YaHei UI Light',14,'bold')).place(x=60,y=120)
                            
                     

                        
                        Label(screen1,text='WELCOME TO THE BANK LOCKER SYSTEM',fg='#57a1f8',font=('Microsoft YaHei UI Light',20,'bold')).place(x=50,y=50)
                        
                        Button(screen1,width=20,pady=7,text='Locker 1',bg='#57a1f8',fg='white',border=2,command=cash_withdrawl).place(x=380,y=170)

                        Button(screen1,width=20,pady=7,text='Locker 2',bg='#57a1f8',fg='white',border=2,command=cash_deposite).place(x=380,y=240)

                        # Button(screen1,width=20,pady=7,text='Total Balance',bg='#57a1f8',fg='white',border=2,command=total_bal).place(x=380,y=300)


                    else:
                        messagebox.showerror("Wrong_OTP","Please Enter Valid OTP")
                        print("Invalid Otp")

                
                
                Label(screen,text='USERNAME:').place(x=20,y=150)
                user = Entry(screen,width=25,fg='black',font=('Microsoft YaHei UI Light',11,'bold'))
                user.place(x=120,y=150)

                Label(screen,text='LOCKER PIN:').place(x=20,y=200)
                atmpin = Entry(screen,width=25,fg='black',font=('Microsoft YaHei UI Light',11,'bold'))
                atmpin.place(x=120,y=200)

                Label(screen,text='ENTER OTP:').place(x=20,y=280)
                otp = Entry(screen,width=25,fg='black',font=('Microsoft YaHei UI Light',11,'bold'))
                otp.place(x=120,y=280)

                Button(screen,width=20,pady=7,text='GENERATE OTP',bg='#57a1f8',fg='white',border=2,font=('Microsoft YaHei UI Light',11,'bold'),command=user_details).place(x=380,y=170)

                Button(screen,width=20,pady=7,text='VERIFY OTP',bg='#57a1f8',fg='white',border=2,font=('Microsoft YaHei UI Light',11,'bold'),command=verify_otp).place(x=380,y=260)

                # Button(screen,width=20,pady=7,text='Resend OTP',bg='#57a1f8',fg='white',border=1).place(x=200,y=320)

            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
                messagebox.showerror("Unknown","Person Not Recognised")
            
            cv2.putText(
                        img, 
                        str(id), 
                        (x+5,y-5), 
                        font, 
                        1, 
                        (255,255,255), 
                        2
                    )
            cv2.putText(
                        img, 
                        str(confidence), 
                        (x+5,y+h-5), 
                        font, 
                        1, 
                        (255,255,0), 
                        1
                    )  
        
        cv2.imshow('camera',img) 
        cv2.waitKey(0) & 0xff # Press 'ESC' for exiting video
        # if k == 27:
        #     break
    # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
     
                
      
            
            

#==============================================    
def signup():
    
    # screen1=Toplevel(root)
    # screen1.title("Dashboard1")
    # screen1.geometry('600x500+300+200')
    # screen1.config(bg='white')
    def signup1():
        username=user.get()
        accountno1=accountno.get()
        atmpin1=atmpin.get()
        mobileno1=mobileno.get()
        email_id=email.get()
        address=address1.get()

        if username=='' or accountno1=='' or atmpin1=='' or mobileno1=='' or email_id =='' or address=='':
            messagebox.showerror("empty","empty fields are not allowed")

        
        else:
            conn=sqlite3.connect('ATM_System.db')
            with conn:
               cur=conn.cursor()
               cur.execute('CREATE TABLE IF NOT EXISTS ATM (user TEXT, accountno INT, atmpin INT, mobileno INT, email INT, address1 TEXT)')
               

            cur.execute('INSERT INTO ATM (user,accountno,atmpin,mobileno,email,address1) VALUES(?,?,?,?,?,?)',(username,accountno1,atmpin1,mobileno1,email_id,address))
            conn.commit()
                
            
            cap = cv2.VideoCapture(0)
            cap.set(3,640) # set Width
            cap.set(4,480) # set Height
            faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            face_id = input('\n enter user id end press <return> ==>  ')
            print("\n [INFO] Initializing face capture. Look the camera and wait ...")
            # Initialize individual sampling face count
            count = 0
            while True:
                ret, img = cap.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale( gray, 1.3, 5 )
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    count += 1
                    # Save the captured image into the datasets folder
                    cv2.imwrite("dataset/User." + str(face_id) + '.' +  
                    str(count) + ".jpg", gray[y:y+h,x:x+w])
                    cv2.imshow('image', img)
                k = cv2.waitKey(100) & 0xff
                if k == 27: # press 'ESC' to quit
                    break
                elif count >= 30: # Take 30 face sample and stop video
                    break
            

            # Path for face image database
            path = 'dataset'
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
            # function to get the images and label data
            def getImagesAndLabels(path):
                imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
                faceSamples=[]
                ids = []
                for imagePath in imagePaths:
                    PIL_img = Image.open(imagePath).convert('L') # grayscale
                    img_numpy = np.array(PIL_img,'uint8')
                    id = int(os.path.split(imagePath)[-1].split(".")[1])
                    faces = detector.detectMultiScale(img_numpy)
                    for (x,y,w,h) in faces:
                        faceSamples.append(img_numpy[y:y+h,x:x+w])
                        ids.append(id)
                return faceSamples,ids
            print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
            faces,ids = getImagesAndLabels(path)
            recognizer.train(faces, np.array(ids))
            # Save the model into trainer/trainer.yml
            recognizer.write('TrainingImageLabel/trainer.yml') 
            # Print the numer of faces trained and end program
            print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

            cap.release()
            cv2.destroyAllWindows()
                     
        frame.destroy()
            
   

       # =================REGISTRATION FRAME========================    

    frame=Frame(root,width=400,height=400)
    frame.place(anchor='center', relx=0.5, rely=0.5)

    heading=Label(frame,text='REGISTRATION',fg='#57a1f8',font=('Microsoft YaHei UI Light',25,'bold'))
    heading.place(x=100,y=10)
     
    label = Label(frame,text='Username:',font=("bold")).place(x=40,y=80)
    user = Entry(frame,width=25,fg='black',border=1,bg="white",font=('Microsoft YaHei UI Light',11))
    user.place(x=140,y=80)
    
    label = Label(frame,text='Account_No:',font=("bold")).place(x=40,y=120)
    accountno = Entry(frame,width=25,fg='black',border=1,bg="white",font=('Microsoft YaHei UI Light',11))
    accountno.place(x=140,y=120)
    
    label = Label(frame,text='Locker_Pin:',font=("bold")).place(x=40,y=160)
    atmpin = Entry(frame,width=25,fg='black',border=1,bg="white",font=('Microsoft YaHei UI Light',11))
    atmpin.place(x=140,y=160)

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
img = ImageTk.PhotoImage(Image.open("atm\\l222.png"))

# Create a Label Widget to display the text or Image
label = Label(frame,width=100,height=100, image = img)
label.place(x=150,y=90)

heading=Label(frame,text='LOGIN',fg='#57a1f8',font=('Microsoft YaHei UI Light',40,'bold'))
heading.place(x=120,y=15)


Button(frame,width=20,pady=7,text='Sign in',bg='#57a1f8',fg='white',border=2,font=("bold"),command=signin).place(x=110,y=200)

label=Label(frame,text="Don't have an account",fg='black',bg='white',font=('Microsoft YaHei UI Light',9,'bold'))
label.place(x=120,y=330)

sign_up = Button(frame,width=20,pady=7,text='Sign up',bg='#57a1f8',fg='white',border=2,cursor='hand2',font=("bold"),command=signup).place(x=110,y=275)



root.mainloop()

       
