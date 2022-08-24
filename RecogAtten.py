from tkinter import *
import os
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import pandas as pd
from tkinter import ttk
from PIL import ImageTk
from tkinter import messagebox
import threading
import numpy as np
from keras.models import load_model
#import dlib
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf    
tf.random.set_seed(1234)

#import dlib
import matplotlib.pyplot as plt
import tensorflow as tf    
from datetime import datetime
from IPython.display import display
from datetime import date
import requests
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def listToString(s): 
    
    # initialize an empty string
    str1 = " " 
    
    # return string  
    return (str1.join(s))


def check_if_running_rec(thread, window):
    """Check every second if the function is finished."""
    if thread.is_alive():
        window.after(1000, check_if_running_rec, thread, window)
    else:
        window.destroy()
        messagebox.showinfo("Recognize face and mark attendance","The attendance has been successfully marked")

def openRec():
    recW = Toplevel(app)
 
    # sets the title of the
    # Toplevel widget
    recW.title("Recognize and Mark Attendance")
 
    # sets the geometry of toplevel
    recW.geometry("400x110")
    def step():
        my_progress.start(10)
    Label(recW, text ="Recognizing faces and Marking Attendance").pack()
    my_progress = ttk.Progressbar(recW, orient=HORIZONTAL, length=300, mode='determinate')
    # A Label widget to show in toplevel
    my_progress.pack(pady=20)
    
    step()
    
    thread= threading.Thread(target=Rec)
    
    thread.start()
    recW.after(1000, check_if_running_rec, thread, recW)
    
def Rec():
    
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
    
    data = np.load('embed_train.npz')
    X_train, trainy, X_test, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    
    # Encode the labels
    le = LabelEncoder()
    train_labels = le.fit_transform(trainy)
    test_labels= le.fit_transform(testy)
    num_classes = len(np.unique(train_labels))
    train_labels = train_labels.reshape(-1, 1)
    test_labels = test_labels.reshape(-1, 1)
    
    #Loading Facenet and Classifier
    face_recognise_model=load_model("Face_recogniser.h5")
    facenet_model = load_model('facenet_keras.h5')
    
    def findCosineDistance(vector1, vector2):
        """
        Calculate cosine distance between two vector
        """
        vec1 = vector1.flatten()
        vec2 = vector2.flatten()
        a = np.dot(vec1.T, vec2)
        b = np.dot(vec1.T, vec1)
        c = np.dot(vec2.T, vec2)
        return 1 - (a/(np.sqrt(b)*np.sqrt(c)))
    
    def CosineSimilarity(test_vec, source_vecs):
        """
        Verify the similarity of one vector to group vectors of one class
        """
        cos_dist = 0
        for source_vec in source_vecs:
            cos_dist += findCosineDistance(test_vec, source_vec)
        return cos_dist/len(source_vecs)
    
    def extract_face_from_image(image_path, required_size=(160, 160)):
      # load image and detect faces
        image = plt.imread(image_path)
        detector = MTCNN()
        faces = detector.detect_faces(image)
        face_images = []
        global count
        count=0
        for face in faces:
            
            # extract the bounding box from the requested face
            x1, y1, width, height = face['box']
            x1, y1= abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            
            # extract the face
            face_boundary = image[y1:y2, x1:x2]
    
            # resize pixels to the model size
            face_image = Image.fromarray(face_boundary)
            face_image = face_image.resize(required_size)
            face_array = asarray(face_image)
            face_images.append(face_array)
            count=count+1
            global array
            array= face_array
        return face_images
    
    data = np.load('embed_train.npz')
    X_train, trainy, X_test, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    
    model=load_model("Face_recogniser.h5")
    detector=MTCNN()
    
    today = datetime.now().strftime("%Y-%m-%d")
    Copy_to_path="Daily Photos/"
    if not os.path.exists(Copy_to_path):
        os.makedirs(Copy_to_path)
    with open("IPinfo.ini") as file:
        for line in file:
            urls = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[0-9]+(?:\.[0-9]+){3}:[0-9]+', line)
            urls = listToString(urls)
            print(urls)
    im = Image.open(requests.get(urls, stream=True).raw)
    im.show()
    im.save(Copy_to_path+today+'.jpeg')
    
    extracted_face = extract_face_from_image(Copy_to_path+today+'.jpeg')
    known_face=[]
    kcount=0
    ucount=0
    embedded_dim=128
    for i in range(count):
        image=extracted_face[i]
        images = image
        confidence_scores=[]
        thumbnail = images.astype('float32')
        aligned_thumbnail=thumbnail.copy()
        face_img=thumbnail
        mean, std = thumbnail.mean(), thumbnail.std()
        face_img = (thumbnail-mean)/std
        vector = facenet_model.predict(np.expand_dims(face_img, axis=0))[0]
        # Predict class
        preds = model.predict(vector.reshape(1,-1))
        preds = preds.flatten()
        # Get the highest accuracy embedded vector
        j = np.argmax(preds)
        proba = preds[j]
        match_class_idx=np.where(trainy==le.classes_[j])[0]
        match_class_idxs = np.where(match_class_idx)[0]
        selected_idx = np.random.choice(match_class_idx, 5)
        compare_embeddings = X_train[selected_idx]
        # Calculate cosine similarity
        cos_similarity = CosineSimilarity(vector, compare_embeddings)
        text = "Unknown"
        if cos_similarity < 0.50 and proba > 0.90:
            name = le.classes_[j]
            known_face.append(name)
            print("Present: {} <{:.2f}>".format(name, proba*100))
            kcount=kcount+1
        else:
            ucount=ucount+1
            
    today = date.today()
    strToday= str(today)
    df= pd.read_csv('attendance.csv')
    df.set_index('names', inplace=True)
    if strToday in df:
        for i in range(kcount):
            df.at[known_face[i],strToday]=1
    else:
        df[strToday]=0
        for i in range(kcount):
            df.at[known_face[i],strToday]=1
    
    print("No. of unknown faces in the photograph: ",ucount)
    display(df)
    df.to_csv('attendance.csv')

def onClick():
    messagebox.showinfo("Message",  "Your IP for photo sharing has been configured successflly")

def save_text():
    text_save=open('IPinfo.ini','w')
    # text_save.write(text_box.get(1.0,END))
    # text_save.close()
    data = str(text_box.get(1.0, END))
    text_save.write(data)
   
    text_save.close()

def open_text():
    text_file= open("IPinfo.ini","r")
    read= text_file.read()
    text_box.insert(END, read)
    text_file.close()

def open_att():
    path = 'attendance.csv'
    os.startfile(path)
    

app= Tk()

myimg= ImageTk.PhotoImage(Image.open("appBackground.jpg"))
label=Label(image=myimg)
label.pack()
app.title('Attendance using Face Recognition: Recognition Phase')
app.geometry('435x516')
app.configure(background="#3B3B3B")
text_box=Text(app,width=50,height=0.5, font=('Helvetica',16))
text_box.pack(pady=20,padx=40)

if os.path.isfile('IPinfo.ini'):
    open_text()

save = Button(app,text="Configure", command=lambda:[save_text(),onClick()])
save.pack(padx=5,pady=5)
recBtn = Button(app, text='Recognize and Record Attendance', width=30,bg="#4aba3f",fg="white", command=openRec)
recBtn.pack(pady=50, padx=40)
openCsv= Button(app, text='Open Attendance Sheet',width=30,bg="#4aba3f",fg="white", command=open_att)
openCsv.pack(pady=0,padx=0)
app.mainloop()
