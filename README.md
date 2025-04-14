#  Currency Authenticity Detector

A machine learning-based web application that detects whether a currency note is **genuine** or **counterfeit** using image classification with a Convolutional Neural Network (CNN). Built with TensorFlow, Keras, OpenCV, and Streamlit.


##  Features

- Upload image of a currency note
- Predicts whether the note is **genuine** or **fake**
- Confidence score displayed with styled results
- CNN model trained from scratch using Keras
- Responsive and interactive frontend built using Streamlit

## How It Works

1. The CNN model is trained using hundreds of real and fake currency images.
2. It learns key visual features that differentiate authentic notes from counterfeit ones.
3. Once trained, the model is saved in `.keras` format.
4. The `Streamlit` frontend allows users to upload an image, which is processed and passed to the model for prediction.
5. The app shows whether the note is real or fake, with a confidence score and supporting details.


## Sample Prediction Flow

1. Upload an image (`.jpg`, `.jpeg`, or `.png`) of a currency note.
2. App resizes the image to `224x224`, normalizes it, and feeds it to the model.
3. The model returns a confidence score.
4. A styled box displays whether the note is real or counterfeit.


## Installation

#### Clone the repository
```bash
git clone https://github.com/aadyatalreja/counterfeit-currency-checker.git
cd counterfeit-currency-checker
```

#### Install required packages
```bash
pip install -r requirements.txt
```
#### Launch the web app
```bash
streamlit run app.py
```

## Dataset
The dataset should be organized as follows:
```go
/IndianCurrencyDataset/
├── train/
│   ├── real/
│   └── fake/
├── validation/
│   ├── real/
│   └── fake/
├── test/
    ├── real/
    └── fake/
```
## Disclaimer
This tool is for educational purposes only. It is not intended for commercial or financial decision-making. Always verify results with official sources or experts.


