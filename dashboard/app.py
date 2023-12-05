import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2
import joblib

feature_list = np.array(pickle.load(open("src/embeddings.pkl",'rb')))
filenames = pickle.load(open('src/filename3.pkl','rb'))
mean_vals = joblib.load("src/mean_vals3.pkl")

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def extract_features(img_path,model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (80,60))
    img = img.flatten() - mean_vals
    embeded_img = pca.transform([img])
    recon_img = pca.inverse_transform(embeded_img)[0].reshape(80,60,3)
    img1 = cv2.resize(recon_img, (224,224))
    expanded_img_array = np.expand_dims(img1,axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

pca = joblib.load("src/pca3.pkl")
# steps
# file upload -> save
uploaded_file = st.file_uploader(label="Choose an image",type=["png", "jpeg", "jpg"])

# Usage
# Create logic for app flow
if not uploaded_file:
    st.warning("Please upload an image.")
    st.stop()
else:
    # Specify the folder where you want to save the uploaded file
    upload_folder = "uploaded/input"
    # Create the folder if it doesn't exist
    os.makedirs(upload_folder, exist_ok=True)
    # Save the uploaded file to the specified folder
    file_path = os.path.join(upload_folder, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.read())
    uploaded_image = uploaded_file.read()
    image1 = Image.open(file_path)
    st.image(image1.resize((200,224)))
    pred_button = st.button("Predict")

def show_image(filename):
    image1 = Image.open(filename).resize((224,224))
    return image1
# Did the user press the predict button?
if pred_button:
    pred_button = True
else:
    st.warning("press predict button") 

# And if they did...
if pred_button:    
    up_image = Image.open(uploaded_file)
    features = extract_features(file_path,model)
    indices = recommend(features,feature_list)
    # show
    col1,col2,col3,col4,col5 = st.columns(5)

    with col1:
        st.image(show_image(filenames[indices[0][5]]))
    with col2:
        st.image(show_image(filenames[indices[0][1]]))
    with col3:
        st.image(show_image(filenames[indices[0][2]]))
    with col4:
        st.image(show_image(filenames[indices[0][3]]))
    with col5:
        st.image(show_image(filenames[indices[0][4]]))

