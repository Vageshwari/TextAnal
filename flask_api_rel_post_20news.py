# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:55:03 2019

@author: Shilpa Dhamapurkar
Reference ;https://www.udemy.com/deploy-data-science-nlp-models-with-docker-containers/learn/lecture/10504546#questions/5338980
"""

import pickle
from flask import Flask, request, jsonify
from flasgger import Swagger
import numpy as np
#import pandas as pd
#import nltk.stem
from datetime import datetime
#from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp
from classTifdf import StemmedTfidfVectorizer
#from classTifdf import train_data


app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict_post_file', methods=["POST"])
def predict_post_cluster_file():
    """Example file endpoint returning a cluster prediction of news-post
    ---
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true
    responses:
        200:
          description: OK
          schema:
            properties:
                new_post_label:
                  type: integer
                  description: Cluster number of new post                
                resp1:
                  type: string
                  description: Similar posts indices
                show_at_1:
                  type: string
                  description: 1st Nearest post
                show_at_2:
                  type: string
                  description: 2nd Nearest post
                show_at_3:
                  type: string
                  description: 3rd Nearest post
    """
    #input_data = pd.read_csv(request.files.get("input_file"), header=None)
    filename_km = r"C:\00_Users\sdd\00_GD2\bzn\prj\fima\Src\python\TextAnal\news-post-cluster-model-" +  datetime.now().strftime("%Y%m%d").upper() + '.pkl'
    print(filename_km)
    with open(filename_km, 'rb') as model_file:
        model_km = pickle.load(model_file)

    filename_vect = r"C:\00_Users\sdd\00_GD2\bzn\prj\fima\Src\python\TextAnal\vectorizer-" +  datetime.now().strftime("%Y%m%d").upper() + '.pkl'
    print(filename_vect)
    with open(filename_vect, 'rb') as vect_file:
        vectorizer = pickle.load(vect_file)
    #vectorized = vectorizer.fit_transform(train_data.data)   
    filename_vected = r"C:\00_Users\sdd\00_GD2\bzn\prj\fima\Src\python\TextAnal\vectorized-" +  datetime.now().strftime("%Y%m%d").upper() + '.pkl'
    with open(filename_vected, 'rb') as vected_file:
        vectorized = pickle.load(vected_file)
    filename_data = r"C:\00_Users\sdd\00_GD2\bzn\prj\fima\Src\python\TextAnal\train_data-" +  datetime.now().strftime("%Y%m%d").upper() + '.pkl'
    with open(filename_data, 'rb') as data_file:
        train_data = pickle.load(data_file)
    print (type(train_data))    
    new_post_file = request.files.get("input_file")
    new_post = new_post_file.read()
    #print (new_post)
    print ("Decoded string printed-----------")
    print (str(new_post,"utf-8") )
    print (type(vectorizer))
    #vectorizer._validate_vocabulary()
    new_post_vec = vectorizer.transform([new_post])
    #print (new_post_vec)
    #new_post_vec = new_post_vec.decode("utf-8") 
    new_post_label = model_km.predict(new_post_vec)[0]
    print ("New post cluster:", new_post_label )
    similar_indices = (model_km.labels_ == new_post_label).nonzero()[0]
    #print (similar_indices)
    print (type(similar_indices))
    print (type(str(similar_indices)))
    similar = []
    for i in similar_indices:
        dist = sp.linalg.norm((new_post_vec - vectorized[i]).toarray())
        similar.append((dist, train_data.data[i]))
    
    similar = sorted(similar)
    print("Count similar: %i" % len(similar))
    
    show_at_1 = similar[0]
    show_at_2 = similar[int(len(similar) / 10)]
    show_at_3 = similar[int(len(similar) / 2)]

    resp1 = str(list(similar_indices))
    #return {'new_post_label':new_post_label, 'resp1': resp1}
    return jsonify({'new_post_label':int(new_post_label), 'resp1': resp1, 'show_at_1':show_at_1, 'show_at_2':show_at_2, 'show_at_3':show_at_3})

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000)
    
    
