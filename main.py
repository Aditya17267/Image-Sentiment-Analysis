import streamlit as st
from PIL import Image
import cv2
import numpy as np
from tensorflow import keras

model = keras.models.load_model("best_model.h5")

def img_data(d): 
  tx = []
  img = cv2.resize(d,(224,224) ,interpolation = cv2.INTER_AREA)
  tx.append(img)
  tx = np.array(tx)
  return tx

def sentiment(p):
  if(p == 0):
    return 'Negative'
  else:
    return "Positive"

def get_prediction(x):
  tx = img_data(x)
  res = np.argmax(model.predict(tx))
  return sentiment(res)

st.title("Image Sentiment Analysis")

st.subheader("Taking image from the user " +
    "and predicting whether" + 
    " the image is positive or negative.")

image = st.file_uploader("Insert the picture: ")

original = Image.open(image)
image = Image.open(image)
img_array = np.array(image)
st.image(original)

prediction = get_prediction(img_array)

st.subheader("Model Prediction")
st.write(prediction)