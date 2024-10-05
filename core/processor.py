#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2 as cv
import numpy as np
import pandas as pd


# In[4]:


import tensorflow as tf
import tensorflow_hub as hub


# In[5]:


from sklearn.preprocessing import StandardScaler


# In[6]:


from sklearn.cluster import KMeans 


# In[7]:


from sklearn.metrics import silhouette_score


# In[8]:


import os


# In[19]:


def get_faces(frame):
    # converting to grayscale
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # detecting faces
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5)

    frame_faces = []
    for (x, y, w, d) in faces:
        frame_faces.append(frame[x: x+w, y: y+d])

    return frame_faces, faces


# In[20]:


def get_features(frame, model):
    embeddings = model(frame)
    return embeddings.numpy() 


# In[21]:


def get_images_processed(directory, model):
    
    # Function to get all files from the directory
    def get_list():
        return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    paths = get_list()  # Get all file paths in the directory

    tags = {}  # Dictionary to hold image file names and corresponding features
    dataset = []  # List to hold all feature vectors

    for path in paths:
        com_path = os.path.join(directory, path)  # Complete path to the file
        image = cv.imread(com_path)  # Read the image

        frame_faces, faces = get_face(image)  # Get faces and face regions from the image
        
        tags[path] = 0  # Initialize the list for the current image

        for f in frame_faces:
            # Extract features for the current face
            features = get_features(f, model).tolist()
            tags[path] += 1  # Store features in tags
            dataset.append(features)  # Store features in the dataset

    return tags, dataset


# In[22]:


def clustering(dataset, tags, max_clusters=20, threshold=0.01):
    X = numpy.array(dataset)

    best_kmean=None
    n=2
    max_score=-1
    for i in range(n, max_clusters+1):
        kmean = KMeans(n_clusters=i)
        labels = kmean.fit_predict(X)
        score = silhouette_score(X, labels)
        prev_score = -1

        if score > max_score:
            max_score = score
            n = i
            best_kmean = kmean
        # early stopping
        if abs(score - prev_score) < threshold:
            break

        prev_score = score

    
    labels_list = best_kmeans.labels_.tolist()
    tagged = {}
    track = 0
    for i, num_faces in tags.items():
        tagged[i] = []
        for j in range(track, track+num_faces):
            tagged.append(labels_list[j])
        track += num_faces

    return tagged, n, best_kmeans


# In[ ]:




