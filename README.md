# Image Sentiment Analysis

This project aims to predict the sentiment (positive or negative) of images using a pre-trained convolutional neural network (VGG19) model. It involves training a model, building a frontend with Streamlit for user-uploaded image sentiment analysis, and offering a Flask-based API for similar predictions.

## Features

- **Sentiment Prediction**: The model classifies images into positive or negative categories.
- **Model**: Pre-trained VGG19 model (using ImageNet weights) fine-tuned for sentiment analysis.
- **User Interface**:
  - **Streamlit app**: Allows users to upload images and get real-time sentiment predictions.
  - **Flask app**: API endpoint for uploading images and receiving sentiment predictions with confidence scores.

## Technologies Used

- **Python**
- **TensorFlow & Keras** (for model creation and training)
- **OpenCV** (for image preprocessing)
- **Streamlit** (for frontend)
- **Flask** (for API)
- **Pandas & NumPy** (for data handling and manipulation)

## Model Training

The model is trained on a dataset of images with labeled sentiments using a pre-trained VGG19 architecture. The following are the key steps:

1.  **Dataset Preprocessing**:

    - The dataset contains image URLs and sentiment labels.
    - The dataset is truncated and shuffled for efficient training.
    - Images are fetched from the URLs, resized to 224x224, and converted into NumPy arrays.

2.  **Model Architecture**:

    - VGG19 (pre-trained on ImageNet) is used as the base model.
    - A custom classification head consisting of a Global Average Pooling layer, a Dense layer, and a Dropout layer is added.
    - The model predicts whether the image sentiment is **positive** or **negative**.

3.  **Training**:

    - The model is trained using `binary_crossentropy` loss with `Adam` optimizer.
    - Training and validation data are split for performance evaluation.

## Usage

### 1\. Streamlit App

The Streamlit app provides a graphical interface for users to upload an image and get a sentiment prediction.

To run the Streamlit app:

```bash
streamlit run main.py
```

- **Input**: User-uploaded image.
- **Output**: Predicted sentiment (positive or negative) displayed on the webpage.

### 2\. Flask App

The Flask app exposes an endpoint where users can upload an image, and the model returns the sentiment prediction and confidence value.

To run the Flask app:

```bash
python app.py
```

- **Input**: Image uploaded via an HTML form.
- **Output**: Sentiment prediction with confidence score displayed on the webpage.

## File Structure

```bash
Image-Sentiment-Analysis/
│
├── model_training.py     # Script to train and save the model
├── main.py               # Streamlit app
├── app.py                # Flask app
├── templates/            # HTML templates for Flask app
│   ├── index.html        # Upload page
│   └── output.html       # Output page
├── best_model.h5         # Trained model file
└── README.md             # Project documentation
```

## How to Run

### Prerequisites

- Python 3.7+
- TensorFlow, Keras
- OpenCV
- Streamlit
- Flask
- Other necessary packages in `requirements.txt` (if any)

### Steps to Run the Project

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/Image-Sentiment-Analysis.git

    cd Image-Sentiment-Analysis
    ```

2.  **Install dependencies**:

    ```bash
    `pip install -r requirements.txt`
    ```

3.  **Run the Streamlit app**:

    ```bash
    `streamlit run main.py`
    ```

4.  **Run the Flask app**:

    ```bash
    python app.py
    ```

## Model Usage

### Prediction Function

- The model predicts sentiment as follows:

```python
def get_prediction(x):
  tx = img_data(x)
  res = np.argmax(model.predict(tx))
  return sentiment(res)`
```

- **Sentiment Function**: The sentiment is determined based on the predicted output:

```python
def sentiment(p):
  if p == 0:
    return 'Negative'
  else:
    return 'Positive'`
```

## Future Scope

- **Enhanced Text Extraction Techniques**: Incorporate OCR for analyzing text sentiment along with images.
- **User Interface Improvements**: Build a more robust UI for better user interaction.
- **Error Handling**: Improve error handling in image fetching and prediction.
- **Multilingual Support**: Expand to analyze images with text in different languages.
- **Security Enhancements**: Secure image uploads and prediction APIs.
