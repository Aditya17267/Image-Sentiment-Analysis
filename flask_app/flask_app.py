from flask import Flask, request, render_template
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("../best_model.h5")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    image = Image.open(request.files['image'].stream)
    # Preprocess the image
    img_array = np.array(image)
    tx = []
    img = cv2.resize(img_array,(224,224) ,interpolation = cv2.INTER_AREA)
    tx.append(img)
    tx = np.array(tx)
    # Use the model to make a prediction
    p = model.predict(tx)
    res = np.argmax(p)
    val = round((p[0][res]) * 10, 1)
    prediction = sentiment(res)
    # Return the prediction to the user
    return render_template("output.html", prediction=prediction,val=val)

def sentiment(p):
  if(p == 0):
    return 'Negative'
  else:
    return "Positive"

if __name__ == '__main__':
    app.run(debug=True)