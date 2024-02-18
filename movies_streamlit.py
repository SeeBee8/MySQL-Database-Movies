import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from lime.lime_text import LimeTextExplainer

import json
with open('config/filepaths.json') as f:
    FPATHS = json.load(f)
st.title("Predicting Yelp Review Ratings")


# Get text to predict from the text input box
X_to_pred = st.text_input("### Enter text to predict here:", 
                          value="Best movie ever!")

# Loading the ML model
@st.cache_resource
def load_ml_model(fpath):
    loaded_model = joblib.load(fpath)
    return loaded_model
# Load model from FPATHS
model_fpath = FPATHS['models']['rf']
clf_pipe = load_ml_model(model_fpath)

# load target lookup dict
@st.cache_data
def load_lookup(fpath=FPATHS['data']['ml']['target_lookup']):
    return joblib.load(fpath)
@st.cache_resource
def load_encoder(fpath=FPATHS['data']['ml']['label_encoder'] ):
    return joblib.load(fpath)

# Load the target lookup dictionary
target_lookup = load_lookup()
target_lookup

# Load the encoder
encoder = load_encoder()
encoder.classes_

# Basic Function to obtain prediction
def make_prediction(X_to_pred, clf_pipe=clf_pipe, lookup_dict= target_lookup):
    # Get Prediction
    pred_class = clf_pipe.predict([X_to_pred])[0]
    # Decode label
    pred_class = lookup_dict[pred_class]
    return pred_class
# Trigger prediction with a button
if st.button("Get prediction."):
    pred_class = make_prediction(X_to_pred)
    st.markdown(f"##### Predicted category:  {pred_class}") 
else: 
    st.empty()


@st.cache_resource
def get_explainer(class_names = None):
    lime_explainer = LimeTextExplainer(class_names=class_names)
    return lime_explainer
    
def explain_instance(explainer, X_to_pred, predict_func):
    explanation = explainer.explain_instance(X_to_pred, predict_func)
    return explanation.as_html(predict_proba=False)
# Create the lime explainer
explainer = get_explainer(class_names = encoder.classes_)

## Loading our training and test data
@st.cache_data
def load_Xy_data(joblib_fpath):
    return joblib.load(joblib_fpath)
# Load training data from FPATHS
train_data_fpath  = FPATHS['data']['ml']['train']
X_train, y_train = load_Xy_data(train_data_fpath)
# Load test data from FPATHS
test_data_fpath  = FPATHS['data']['ml']['test']
X_test, y_test = load_Xy_data(test_data_fpath)

##To place the 3 checkboxes side-by-side
col1,col2,col3 = st.columns(3)
show_train = col1.checkbox("Show training data.", value=True)
show_test = col2.checkbox("Show test data.", value=True)
show_model_params =col3.checkbox("Show model params.", value=False)
if st.button("Show model evaluation."):
    pass # placeholder



if st.button("Show model evaluation."):

    if show_train == True:
    # Display training data results
    y_pred_train = clf_pipe.predict(X_train)
    report_str, conf_mat = classification_metrics_streamlit(y_train, y_pred_train,
                                                            label='Training Data')
    st.text(report_str)
    st.pyplot(conf_mat)
    st.text("\n\n")
    if show_test == True: 
    # Display the trainin data resultsg
    y_pred_test = clf_pipe.predict(X_test)
    report_str, conf_mat = classification_metrics_streamlit(y_test, y_pred_test,
                                                            cmap='Reds',label='TestData')
    st.text(report_str)
    st.pyplot(conf_mat)
    st.text("\n\n")
    
    if show_model_params:
    # Display model params
    st.markdown("####  Model Parameters:")
    st.write(clf_pipe.get_params())
else: st.empty()



