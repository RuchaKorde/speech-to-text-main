import streamlit as st
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from io import BytesIO

# Load the pre-trained Wav2Vec2 model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Function to transcribe audio
def transcribe_audio(audio_data):
    try:
        # Load the audio file with librosa
        audio, rate = librosa.load(BytesIO(audio_data), sr=16000)
        
        # Prepare the input values
        input_values = tokenizer(audio, return_tensors="pt").input_values

        # Perform the forward pass and get the logits
        logits = model(input_values).logits

        # Get the predicted IDs and decode them to text
        prediction = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(prediction)[0]
        
        return transcription
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit UI
st.set_page_config(page_title="Audio Transcription with Wav2Vec2", layout="wide")

# Custom header with an image
st.markdown("""
    <style>
    .header {
        background: #0072b1;
        padding: 20px;
        text-align: center;
        color: white;
        font-size: 28px;
        font-weight: bold;
        border-radius: 8px;
    }
    .main {
        background: #f0f2f6;
        padding: 20px;
        border-radius: 8px;
    }
    .stButton>button {
        background-color: #0072b1;
        color: white;
        border: None;
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #005a9e;
    }
    .file-info {
        margin-top: 20px;
        font-size: 18px;
    }
    .transcription {
        background: #ffffff;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #ddd;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header">🎙️ Audio Transcription with Wav2Vec2</div>', unsafe_allow_html=True)

# Main Content
st.markdown("""
    <div class="main">
        <p><strong>Welcome to the Audio Transcription App!</strong></p>
        <p>Upload a <code>.wav</code> audio file and get a transcription using the Wav2Vec2 model. Supported formats: WAV.</p>
    </div>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    # Display file information
    st.markdown(f"""
        <div class="file-info">
            <p><strong>File Name:</strong> {uploaded_file.name}</p>
            <p><strong>File Size:</strong> {uploaded_file.size / 1024:.2f} KB</p>
        </div>
    """, unsafe_allow_html=True)

    # Display audio player
    st.audio(uploaded_file, format="audio/wav")

    # Show a progress spinner while processing
    with st.spinner("Transcribing audio..."):
        # Transcribe the audio
        audio_data = uploaded_file.read()
        transcription = transcribe_audio(audio_data)
    
    # Display the transcription result
    st.markdown(f"""
        <div class="transcription">
            <h2>📜 Transcription</h2>
            <p>{transcription}</p>
        </div>
    """, unsafe_allow_html=True)
