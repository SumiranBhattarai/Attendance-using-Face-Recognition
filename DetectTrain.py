from tkinter import *
import os
from os import listdir
from os.path import isdir
from PIL import Image
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from pathlib import Path
import pandas as pd
from tkinter import ttk
from PIL import ImageTk
from tkinter import messagebox

from PIL import ImageEnhance
import cv2
import threading
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
from keras.models import load_model
#import dlib
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf    
tf.random.set_seed(1234)

#import dlib
import tensorflow as tf    


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def check_if_running_var(thread, window):
    """Check every second if the function is finished."""
    if thread.is_alive():
        window.after(1000, check_if_running_var, thread, window)
    else:
        window.destroy()
        messagebox.showinfo("Data Augmentation","Variations Successfully generated")      

def check_if_running_det(thread, window):
    """Check every second if the function is finished."""
    if thread.is_alive():
        window.after(1000, check_if_running_det, thread, window)
    else:
        window.destroy()
        messagebox.showinfo("Detect Face","Detection of Face has been done") 
        
def check_if_running_emd(thread, window):
    """Check every second if the function is finished."""
    if thread.is_alive():
        window.after(1000, check_if_running_emd, thread, window)
    else:
        window.destroy()
        messagebox.showinfo("Embedding","Embedding is successfully extracted and trained") 

def openDetect():
     
    # Toplevel object which will
    # be treated as a new window
    detectW = Toplevel(app)
 
    # sets the title of the
    # Toplevel widget
    detectW.title("Face Detection")
 
    # sets the geometry of toplevel
    detectW.geometry("400x110")
    def step():
        my_progress.start(10)
    Label(detectW, text ="Please Wait untill the faces from the dataset is extracted").pack()
    my_progress = ttk.Progressbar(detectW, orient=HORIZONTAL, length=300, mode='determinate')
    # A Label widget to show in toplevel
    my_progress.pack(pady=20)
    
    step()
    
    thread= threading.Thread(target=Detect)
    
    thread.start()
    detectW.after(1000, check_if_running_det, thread, detectW)

def Detect():

    
    persons_dataset_path='programData/images/'
    
    def extract_face(path_to_filename, detector, required_size=(160,160),save_faces=False):
        image= Image.open(path_to_filename)
        image= image.convert('RGB')
        pixels=asarray(image)
        results=detector.detect_faces(pixels)
        x1,y1,width,height=results[0]['box']
        x1,y1=abs(x1),abs(y1)
        x2,y2 = x1 + width, y1 + height
        face= pixels[y1:y2 , x1:x2]
        image=Image.fromarray(face)
        image=image.resize(required_size)
        if (save_faces):
            path= os.path.split(os.path.abspath(path_to_filename))[0]
            file_name= os.path.split(os.path.abspath(path_to_filename))[1]
            person_name= os.path.basename(os.path.normpath(Path(path)))
            project_folder= Path(path).parent.parent
            print(person_name)
            target_folder= os.path.join(project_folder, 'faces_mini_dataset',person_name)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            target_face_file_path= os.path.join(target_folder, file_name)
            print(target_face_file_path)
            image.save(target_face_file_path)
        face_array= asarray(image)
        return face_array
    
    def extract_faces(directory):
        print('load_faces')
        faces=list()
        
        detector=MTCNN()
        print('extracting faces from',directory,'....')
        
        for filename in listdir(directory):
            path= directory+ filename
            try:
                face= extract_face(path, detector, save_faces=True)
            except Exception as e:
                continue
           
            #face= extract_face(path, detector, save_faces=True)
            faces.append(face)
        return faces
    
    def generate_faces_from_images(directory):
        print('load dataset...')
        x,y =list(),list()
        num=1
        for subdir in listdir(directory):
            
            path=directory+ subdir +'/'
            if not isdir(path):
                continue
            
            faces= extract_faces(path)
            labels = [subdir for _ in range(len(faces))]
            
            print('>%d) loaded %d examples for class %s' % (num,len(faces),subdir))
            num+=1
            x.extend(faces)
            y.extend(labels)
        return asarray(x), asarray(y)
    
    '''
    def generate_excel(directory):
        names=[]
        roll_no=[]
        for subdir in listdir(directory):
            path=subdir
            print("Enter Roll No for "+os.path.basename(path))
            p=input()
            names.append(os.path.basename(path))
            roll_no.append(p) 
        df = pd.DataFrame({'roll_no':roll_no, 'names':names})
        df.to_csv('attendance.csv', index=False)
    
    
    generate_excel(persons_dataset_path)
    '''
    faces, labels= generate_faces_from_images(persons_dataset_path)
    print(faces.shape, labels.shape)
    
    savez_compressed("face_dataset_numpy.npz", faces, labels)



def openVariations():

    varW = Toplevel(app)
 
    # sets the title of the
    # Toplevel widget
    varW.title("Data Augmentation")
 
    # sets the geometry of toplevel
    varW.geometry("400x110")
    def step():
        my_progress.start(10)
    Label(varW, text ="Please Wait untill the generation of variations is taking place").pack()
    my_progress = ttk.Progressbar(varW, orient=HORIZONTAL, length=300, mode='determinate')
    # A Label widget to show in toplevel
    my_progress.pack(pady=20)
    
    step()
    
    thread= threading.Thread(target=Variations)
    
    thread.start()
    varW.after(1000, check_if_running_var, thread, varW)
    
def Variations():
    def load_faces(direct, directory):
        for filename in listdir(direct):
            fileName= direct+filename
            
            try:
                #saturation(fileName,directory)
                brightnessAndContrast(fileName, directory)
                transformation(fileName, directory)
            
            except Exception as e:
                continue
        
    def load_dataset(directory, new_path):
        for subdir in listdir(directory):
            path= directory+subdir+'/'
            new = new_path + subdir + '/'
            if os.path.isdir(new) == False:
                os.makedirs(new)
            print("Generating variations for "+subdir)
            if not isdir(path):
                continue
            load_faces(path, new)   
            
    def brightnessAndContrast(fileName, directory):
        img= Image.open(fileName)
        enhancer= ImageEnhance.Brightness(img)
        brightnessImage= enhancer.enhance(2)
        brightnessImage.save(directory+"Brightness1_Image.jpg")
        
        img= Image.open(fileName)
        enhancer= ImageEnhance.Brightness(img)
        brightnessImage= enhancer.enhance(1.5)
        brightnessImage.save(directory+"Brightness2_Image.jpg")
        
        img= Image.open(fileName)
        enhancer= ImageEnhance.Brightness(img)
        brightnessImage= enhancer.enhance(1)
        brightnessImage.save(directory+"Brightness3_Image.jpg")
        
        img= Image.open(fileName)
        enhancer= ImageEnhance.Brightness(img)
        brightnessImage= enhancer.enhance(0.5)
        brightnessImage.save(directory+"Brightness4_Image.jpg")
        
        img= Image.open(fileName)
        enhancer= ImageEnhance.Contrast(img)
        brightnessImage= enhancer.enhance(2)
        brightnessImage.save(directory+"Contrast1_Image.jpg")
        
        img= Image.open(fileName)
        enhancer= ImageEnhance.Contrast(img)
        brightnessImage= enhancer.enhance(3)
        brightnessImage.save(directory+"Contrast2_Image.jpg")
        
        img= Image.open(fileName)
        enhancer= ImageEnhance.Contrast(img)
        brightnessImage= enhancer.enhance(4)
        brightnessImage.save(directory+"Contrast3_Image.jpg")
        
        img= Image.open(fileName)
        enhancer= ImageEnhance.Contrast(img)
        brightnessImage= enhancer.enhance(5)
        brightnessImage.save(directory+"Contrast4_Image.jpg")
        
        
        
    def transformation(fileName, directory):
        #1
        img= cv2.imread(fileName,1)
        rows,cols=img.shape[:2]
        Matrix= cv2.getRotationMatrix2D((cols/2,rows/2),10,1)
        rotated10=cv2.warpAffine(img,Matrix,(cols,rows))
        cv2.imwrite(directory+'img_rotate_10_left.jpg',rotated10)
        #2
        img= cv2.imread(fileName,1)
        rows,cols=img.shape[:2]
        Matrix= cv2.getRotationMatrix2D((cols/2,rows/2),-10,1)
        rotated10=cv2.warpAffine(img,Matrix,(cols,rows))
        cv2.imwrite(directory+ 'img_rotate_10_right.jpg',rotated10)
        #3
        img= cv2.imread(fileName,1)
        rows,cols=img.shape[:2]
        Matrix= cv2.getRotationMatrix2D((cols/2,rows/2),15,1)
        rotated15=cv2.warpAffine(img,Matrix,(cols,rows))
        cv2.imwrite(directory+ 'img_rotate_15_left.jpg',rotated15)
        #4
        img= cv2.imread(fileName,1)
        rows,cols=img.shape[:2]
        Matrix= cv2.getRotationMatrix2D((cols/2,rows/2),-15,1)
        rotated15=cv2.warpAffine(img,Matrix,(cols,rows))
        cv2.imwrite(directory+ 'img_rotate_15_right.jpg',rotated15)
        #5
        img= cv2.imread(fileName,1)
        rows,cols=img.shape[:2]
        Matrix= cv2.getRotationMatrix2D((cols/2,rows/2),20,1)
        rotated20=cv2.warpAffine(img,Matrix,(cols,rows))
        cv2.imwrite(directory+ 'img_rotate_20_left.jpg',rotated20)
        #6
        img= cv2.imread(fileName,1)
        rows,cols=img.shape[:2]
        Matrix= cv2.getRotationMatrix2D((cols/2,rows/2),-20,1)
        rotated20=cv2.warpAffine(img,Matrix,(cols,rows))
        cv2.imwrite(directory+ 'img_rotate_20_right.jpg',rotated20)
        #7
        img= cv2.imread(fileName, 1)
        img_flip=cv2.flip(img,1)
        cv2.imwrite(directory+ 'img_flip.jpg', img_flip)
        #8
        img= cv2.imread(fileName, 1)
        bilateralblur=cv2.bilateralFilter(img,7,75,75)
        cv2.imwrite(directory+ 'img_Blur_Bilateral_1.jpg', bilateralblur)
        #9
        img= cv2.imread(fileName, 1)
        bilateralblur=cv2.bilateralFilter(img,6,75,75)
        cv2.imwrite(directory+ 'img_Blur_Bilateral_2.jpg', bilateralblur)
        #10
        img= cv2.imread(fileName, 1)
        bilateralblur=cv2.bilateralFilter(img,9,75,75)
        cv2.imwrite(directory+ 'img_Blur_Bilateral_3.jpg', bilateralblur)
        #11
        img= cv2.imread(fileName, 1)
        bilateralblur=cv2.bilateralFilter(img,10,75,75)
        cv2.imwrite(directory+ 'img_Blur_Bilateral_4.jpg', bilateralblur)
        #12
        img= cv2.imread(fileName, 1)
        bilateralblur=cv2.bilateralFilter(img,11,75,75)
        cv2.imwrite(directory+ 'img_Blur_Bilateral_5.jpg', bilateralblur)
        #13
        img= cv2.imread(fileName, 1)
        bilateralblur=cv2.bilateralFilter(img,12,75,75)
        cv2.imwrite(directory+ 'img_Blur_Bilateral_6.jpg', bilateralblur)
        #14
        img= cv2.imread(fileName, 1)
        bilateralblur=cv2.bilateralFilter(img,13,75,75)
        cv2.imwrite(directory+'img_Blur_Bilateral_7.jpg', bilateralblur)
        #15
        img= cv2.imread(fileName, 1)
        bilateralblur=cv2.bilateralFilter(img,14,75,75)
        cv2.imwrite(directory +'img_Blur_Bilateral_8.jpg', bilateralblur)
        #16
        img= cv2.imread(fileName, 1)
        bilateralblur=cv2.bilateralFilter(img,15,75,75)
        cv2.imwrite(directory+'img_Blur_Bilateral_9.jpg', bilateralblur)
        #17
        img= cv2.imread(fileName, 1)
        bilateralblur=cv2.bilateralFilter(img,8,75,75)
        cv2.imwrite(directory+ 'img_Blur_Bilateral_10.jpg', bilateralblur)
        #18
        img= cv2.imread(fileName, 1)
        bilateralblur=cv2.bilateralFilter(img,30,50,20)
        cv2.imwrite(directory+'img_Blur_Bilateral_11.jpg', bilateralblur)
        #19
        img= cv2.imread(fileName, 1)
        gaussianblur= cv2.GaussianBlur(img,(5,5),10000)
        cv2.imwrite(directory+'img_Blur_Gausian.jpg', gaussianblur)
        #20
        img= cv2.imread(fileName, 1)
        medianblur= cv2.medianBlur(img,5)
        cv2.imwrite(directory+ 'img_Blur_median.jpg', medianblur)
        #22
        img= cv2.imread(fileName, 1)
        blueImage=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        cv2.imwrite(directory+ 'img_blue_color.jpg', blueImage)
        #23
        img= cv2.imread(fileName, 1)
        grayImage=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        cv2.imwrite(directory+'img_gray_color.jpg', grayImage)
        #24
        img= cv2.imread(fileName, 1)
        (thresh, blackAndWhiteImage)= cv2.threshold(grayImage,127,255,cv2.THRESH_BINARY)
        cv2.imwrite( directory+'img_BaW_color.jpg', blackAndWhiteImage)
        #25
        img= cv2.imread(fileName, 1)
        yellowImage=cv2.cvtColor(img,cv2.COLOR_XYZ2RGB)
        cv2.imwrite(directory+'img_yellow_color.jpg', yellowImage)
        #26
        img= cv2.imread(fileName, 1)
        colorImage=cv2.cvtColor(img,cv2.COLOR_RGB2XYZ)
        cv2.imwrite( directory+'img_color_1.jpg',colorImage)
        #27
        img= cv2.imread(fileName, 1)
        colorImage=cv2.cvtColor(img,cv2.COLOR_LRGB2Luv)
        cv2.imwrite(directory+'img_color_2.jpg',colorImage)
        #28
        img= cv2.imread(fileName, 1)
        colorImage=cv2.cvtColor(img,cv2.COLOR_XYZ2RGB)
        cv2.imwrite(directory+'img_color_3.jpg',colorImage)
        #29
        img= cv2.imread(fileName, 1)
        blueImage=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        cv2.imwrite(directory+ 'img_blue_color.jpg', blueImage)
        #30
        img= cv2.imread(fileName, 1)
        grayImage=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        cv2.imwrite(directory+'img_gray_color.jpg', grayImage)
        #31
        img= cv2.imread(fileName, 1)
        (thresh, blackAndWhiteImage)= cv2.threshold(grayImage,127,255,cv2.THRESH_BINARY)
        cv2.imwrite(directory+ 'img_BaW_color.jpg', blackAndWhiteImage)
        #32
        img= cv2.imread(fileName, 1)
        yellowImage=cv2.cvtColor(img,cv2.COLOR_XYZ2RGB)
        cv2.imwrite(directory +'img_yellow_color.jpg', yellowImage)
        #33
        img= cv2.imread(fileName, 1)
        colorImage=cv2.cvtColor(img,cv2.COLOR_RGB2XYZ)
        cv2.imwrite(directory+ 'img_color_1.jpg',colorImage)
        #34
        img= cv2.imread(fileName, 1)
        colorImage=cv2.cvtColor(img,cv2.COLOR_LRGB2Luv)
        cv2.imwrite(directory+'img_color_2.jpg',colorImage)
        #35
        img= cv2.imread(fileName, 1)
        colorImage=cv2.cvtColor(img,cv2.COLOR_XYZ2RGB)
        cv2.imwrite(directory+'img_color_3.jpg',colorImage)
        
    
    def saturation(fileName, directory):
        img = cv2.imread(fileName, 1)
        hsvImg=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        
        hsvImg[...,1]= hsvImg[...,1]*2
        
        hsvImg[...,2]= hsvImg[...,2]*0.6
        
        saturation= cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)
        cv2.imwrite(directory+'img_saturation1.jpg', saturation)
        
        
        img = cv2.imread(fileName, 1)
        hsvImg=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        
        hsvImg[...,1]= hsvImg[...,1]*1.7
        
        hsvImg[...,2]= hsvImg[...,2]*0.6
        
        saturation= cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)
        cv2.imwrite(directory+'img_saturation2.jpg', saturation)
        
        
        img = cv2.imread(fileName, 1)
        hsvImg=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        
        hsvImg[...,1]= hsvImg[...,1]*2
        
        hsvImg[...,2]= hsvImg[...,2]*1
        
        saturation= cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)
        cv2.imwrite(directory+'img_saturation3.jpg', saturation)
        
        
        img = cv2.imread(fileName, 1)
        hsvImg=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        
        hsvImg[...,1]= hsvImg[...,1]* -1
        
        hsvImg[...,2]= hsvImg[...,2]*0.6
        
        saturation= cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)
        cv2.imwrite(directory+'img_saturation4.jpg', saturation)
        
        
        img = cv2.imread(fileName, 1)
        hsvImg=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        
        hsvImg[...,1]= hsvImg[...,1]* -0.5
        
        hsvImg[...,2]= hsvImg[...,2]* 0.6
        
        saturation= cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)
        cv2.imwrite(directory+'img_saturation5.jpg', saturation)
        
        
        img = cv2.imread(fileName, 1)
        hsvImg=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        
        hsvImg[...,1]= hsvImg[...,1]* -0.5
        
        hsvImg[...,2]= hsvImg[...,2]*0.6
        
        saturation= cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)
        cv2.imwrite(directory+'img_saturation6.jpg', saturation)
        
        
        
    def generate_excel(directory):
        names=[]
        #roll_no=[]
        for subdir in listdir(directory):
            path=subdir
            #print("Enter Roll No for "+os.path.basename(path))
            #p=input()
            names.append(os.path.basename(path))
            #roll_no.append(p) 
        df = pd.DataFrame({'names':names})
        df.to_csv('attendance.csv', index=False)
    
    
    persons_dataset_path='images/'
    new_path= 'programData/images/'
    if os.path.isdir(new_path) == False:
        os.makedirs(new_path)
    print('Image Augmentation is being performed...')
    generate_excel(persons_dataset_path)
    load_dataset(persons_dataset_path, new_path)
    print("Variation Generated")



def openEmbed():
    embedW = Toplevel(app)
 
    # sets the title of the
    # Toplevel widget
    embedW.title("Embedding Extraction")
 
    # sets the geometry of toplevel
    embedW.geometry("400x110")
    def step():
        my_progress.start(10)
    Label(embedW, text ="Please Wait untill the facial embedding is extracted").pack()
    my_progress = ttk.Progressbar(embedW, orient=HORIZONTAL, length=300, mode='determinate')
    # A Label widget to show in toplevel
    my_progress.pack(pady=20)
    
    step()
    
    thread= threading.Thread(target=Embed)
    
    thread.start()
    embedW.after(1000, check_if_running_emd, thread, embedW)
    
def Embed():
    def get_embedding(model, face):
        # scale pixel values
        face = face.astype('float32')
        # standardization
        mean, std = face.mean(), face.std()
        face = (face-mean)/std
        # transfer face into one sample (3 dimension to 4 dimension)
        sample = np.expand_dims(face, axis=0)
        # make prediction to get embedding
        yhat = model.predict(sample)
        return yhat[0]
    #def create_embeddings(thumbnail_faces):
    model = load_model('facenet_keras.h5')   
    data = np.load('face_dataset_numpy.npz')
    trainX, trainy = data['arr_0'], data['arr_1']
    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]
    for key,train_test_split in enumerate(trainX):
        if key % 2==0:
            X_train.append(trainX[key])
            y_train.append(trainy[key])
        else:
            X_test.append(trainX[key])
            y_test.append(trainy[key])
    emdTrainX = list()
    emdTestX = list()
    try:
        for face in X_train:
            emd = get_embedding(model, face)
            emdTrainX.append(emd)
    
        emdTrainX = np.asarray(emdTrainX)
        for face in X_test:
            emd = get_embedding(model, face)
            emdTestX.append(emd)
        emdTestX = np.asarray(emdTestX)     
    except Exception as err:
        print(err)
    np.savez_compressed('embed_train.npz', emdTrainX, y_train, emdTestX, y_test)
    #return emdTrainX,emdTestX
    #create_embeddings('thumbnail_image')
    data = np.load('embed_train.npz')
    X_train, trainy, X_test, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    
    # Encode the labels
    le = LabelEncoder()
    train_labels = le.fit_transform(trainy)
    test_labels= le.fit_transform(testy)
    num_classes = len(np.unique(train_labels))
    train_labels = train_labels.reshape(-1, 1)
    test_labels = test_labels.reshape(-1, 1)
    #converting clases to categorical variables
    Y_train = np_utils.to_categorical(train_labels, num_classes)
    Y_test = np_utils.to_categorical(test_labels, num_classes)
    input_shape = (128,)
    
    # building a linear stack of layers with the sequential model
    model = Sequential()
    # hidden layer
    model.add(Dense(512, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # looking at the model summary
    model.summary()
    # compiling the sequential model
    # training the model for 30 epochs
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=optimizer,metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=32, epochs=30, validation_data=(X_test, Y_test))
    #saving Model
    model.save('Face_recogniser.h5')

def openHelp():
    helpW = Toplevel(app)
 
    # sets the title of the
    # Toplevel widget
    helpW.title("Data Augmentation")
 
    # sets the geometry of toplevel
    helpW.geometry("400x110")
    def step():
        my_progress.start(10)
    Label(helpW, text ="Please Wait untill the generation of variations is taking place").pack()
    my_progress = ttk.Progressbar(embedW, orient=HORIZONTAL, length=300, mode='determinate')
    # A Label widget to show in toplevel
    my_progress.pack(pady=20)
    
    step()
    
    thread= threading.Thread(target=Embed)
    
    thread.start()
    embedW.after(1000, check_if_running_emd, thread, embedW)

app= Tk()

app.title('Attendance using Face Recognition: Enrollment Phase')
app.geometry('455x477')
app.configure(background="#3B3B3B")

myimg= ImageTk.PhotoImage(Image.open("appBackground.jpg"))
label=Label(image=myimg)
label.pack()

varBtn = Button(app, text='Data Augmentation', width=20,bg="#4aba3f",fg="white", command=openVariations)
varBtn.pack(pady=20, padx=40)

detectBtn = Button(app, text='Face Detection', width=20,bg="#4aba3f",fg="white", command=openDetect)
detectBtn.pack(pady=20, padx=40)

embedBtn = Button(app, text='Embedding Extraction', width=20,bg="#4aba3f",fg="white", command=openEmbed)
embedBtn.pack(pady=20, padx=40)

helpBtn= Button(app,text='Help', width=20,bg="#4aba3f",fg="white", command=openHelp)
app.mainloop()
