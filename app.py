import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

st.set_page_config(
    page_title="Currency Authenticity Detector",
    page_icon="💵",
    layout="centered"
)

st.markdown("""
<style>
    /* Professional Design */
    .stApp {
        background-color: #F8F9FA;
    }
    
    /* Header Styling */
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Sub-header Styling */
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    
    /* Info Box Styling */
    .info-text {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #BFDBFE;
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-weight: 500;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        border: none;
    }
    
    /* Result Box Styling */
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1.5rem;
        text-align: center;
        font-weight: 500;
    }
    
    .real-currency {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    
    .fake-currency {
        background-color: #FEF2F2;
        color: #B91C1C;
        border: 1px solid #FECACA;
    }
    
    /* Upload Area Styling */
    .upload-text {
        color: #1E40AF;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    /* Footer Styling */
    .footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.8rem;
        color: #6B7280;
    }
    
    /* Copyright Styling */
    .copyright {
        text-align: center;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #1E40AF;
    }
    
    /* List styling */
    .tips-list li {
        margin-bottom: 0.75rem;
        list-style-type: none;
    }
    
    .tips-list li:before {
        content: "• ";
        margin-right: 0.5rem;
        color: #2563EB;
    }
    
    /* File uploader styling */
    .css-1offfwp {
        border-radius: 0.5rem;
        border: 1px dashed #93C5FD;
        background-color: #F8F9FA;
        padding: 1rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #EFF6FF;
        border-radius: 0.5rem;
        color: #1E40AF;
        font-size: 1.2rem !important;
        font-weight: 500 !important;
        padding: 0.75rem 1rem !important;
    }
    
    /* Expander content styling */
    .streamlit-expanderContent {
        background-color: #F9FAFB;
        border: 1px solid #DBEAFE;
        border-radius: 0 0 0.5rem 0.5rem;
        padding: 1.25rem !important;
    }
    
    /* Detection details styling */
    .details-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 0.5rem;
    }
    
    .detail-card {
        background-color: #EFF6FF;
        border-radius: 0.5rem;
        padding: 1rem;
        border: 1px solid #BFDBFE;
    }
    
    .detail-label {
        font-weight: 500;
        color: #1E40AF;
        margin-bottom: 0.25rem;
        font-size: 1rem;
    }
    
    .detail-value {
        font-size: 1.1rem;
        color: #1F2937;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #2563EB;
    }
    
    /* Divider styling */
    hr {
        border-top: 1px solid #BFDBFE;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detection_model():
    try:
        return load_model('final-fake-currency.keras')
    except:
        st.error("Error: Model file 'final-fake-currency.keras' not found. Please ensure the model file is in the correct location.")
        return None

model = load_detection_model()

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_currency(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

st.markdown("<h1 class='main-header'>Currency Authenticity Detector</h1>", unsafe_allow_html=True)

st.markdown("""
<div class='info-text'>
    <h3 style='color: #1E40AF; margin-top: 0; font-size: 1.2rem;'>Welcome to the Currency Authenticator</h3>
    <p style='color: #4B5563;'>This advanced application uses machine learning to detect whether a currency note is genuine or counterfeit. Upload a clear image of your currency note for instant analysis.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("<h2 class='sub-header'>Upload Currency Image</h2>", unsafe_allow_html=True)
    st.markdown("<p class='upload-text'>Drop your image here:</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    st.markdown("<p style='color: #6B7280; font-size: 0.8rem;'>Limit 200MB per file • JPG, JPEG, PNG</p>", unsafe_allow_html=True)

with col2:
    st.markdown("<h3 style='color: #1E40AF; margin-top: 1.5rem; font-size: 1.2rem;'>Tips for Best Results</h3>", unsafe_allow_html=True)
    st.markdown("""
    <ul class='tips-list' style='color: #4B5563; padding-left: 0.5rem;'>
        <li>Use a well-lit, clear image</li>
        <li>Position the note to be fully visible</li>
        <li>Avoid shadows and glare</li>
        <li>Place on a contrasting background</li>
    </ul>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Your Currency Image</h2>", unsafe_allow_html=True)
    st.image(image, channels="BGR", use_container_width=True)
    
    analyze_button = st.button('Analyze Currency')
    
    if analyze_button:
        if model is not None:
            st.markdown("<hr>", unsafe_allow_html=True)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.markdown("<p style='color: #1E40AF; text-align: center;'>Preprocessing image...</p>", unsafe_allow_html=True)
            progress_bar.progress(25)
            time.sleep(0.5)
            
            status_text.markdown("<p style='color: #1E40AF; text-align: center;'>Analyzing features...</p>", unsafe_allow_html=True)
            progress_bar.progress(50)
            time.sleep(0.5)
            
            status_text.markdown("<p style='color: #1E40AF; text-align: center;'>Running detection model...</p>", unsafe_allow_html=True)
            progress_bar.progress(75)
            time.sleep(0.5)
            
            prediction = predict_currency(image)
            confidence = prediction[0][0]
            is_fake = confidence > 0.5
            
            status_text.markdown("<p style='color: #1E40AF; text-align: center;'>Finalizing results...</p>", unsafe_allow_html=True)
            progress_bar.progress(100)
            time.sleep(0.5)
            
            status_text.empty()
            progress_bar.empty()
            
            if is_fake:
                st.markdown(f"""
                <div class='result-box fake-currency'>
                    <h2>COUNTERFEIT DETECTED</h2>
                    <p>Confidence: {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-box real-currency'>
                    <h2>GENUINE CURRENCY</h2>
                    <p>Confidence: {(1-confidence):.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with st.expander("View Detection Details"):
                st.markdown("""
                <div class="details-container">
                    <div class="detail-card">
                        <p class="detail-label">Raw Confidence Score</p>
                        <p class="detail-value">{:.4f}</p>
                    </div>
                    <div class="detail-card">
                        <p class="detail-label">Detection Threshold</p>
                        <p class="detail-value">0.5000</p>
                    </div>
                    <div class="detail-card">
                        <p class="detail-label">Model Input Shape</p>
                        <p class="detail-value">224 x 224 pixels</p>
                    </div>
                    <div class="detail-card">
                        <p class="detail-label">Analysis Method</p>
                        <p class="detail-value">Deep Neural Network</p>
                    </div>
                </div>
                """.format(confidence), unsafe_allow_html=True)
                
                # Additional technical information
                st.markdown("""
                <h3 style="color: #1E40AF; margin-top: 1.5rem; font-size: 1.2rem;">Technical Details</h3>
                <div style="background-color: #F9FAFB; border: 1px solid #DBEAFE; border-radius: 0.5rem; padding: 1rem; margin-top: 0.75rem;">
                    <p style="margin-bottom: 0.75rem; color: #4B5563;">
                        <span style="font-weight: 500; color: #1E40AF;">Preprocessing:</span> 
                        The image is resized to 224x224 pixels and normalized to values between 0 and 1.
                    </p>
                    <p style="margin-bottom: 0.75rem; color: #4B5563;">
                        <span style="font-weight: 500; color: #1E40AF;">Interpretation:</span> 
                        Confidence scores above 0.5 indicate a counterfeit note, while scores below 0.5 indicate a genuine note.
                    </p>
                    <p style="color: #4B5563;">
                        <span style="font-weight: 500; color: #1E40AF;">Model Architecture:</span> 
                        This detector uses a convolutional neural network trained on thousands of genuine and counterfeit currency samples.
                    </p>
                </div>
                """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='footer'>This application is for educational purposes only. Results should be verified by experts and should not be relied upon for financial decisions.</div>", unsafe_allow_html=True)

st.markdown("<div class='copyright'>© Aadya Arun Talreja, Aadya Singh, Amogha V Prasad and Archita Das</div>", unsafe_allow_html=True)