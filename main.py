import streamlit as st
import numpy as np
import pandas as pd
import keras
import numpy as np
from scipy import signal as scipy_signal

CLASS_LABELS = [
    "Come", "Stop", "Hurry up", "Crouch", "Rally Point"
]
MODEL = keras.saving.load_model("best_emg_model.keras")
FS = 2000

def extract_envelope(signal):
    analytic_signal = scipy_signal.hilbert(signal)
    envelope = np.abs(analytic_signal)
    return envelope

def preprocess(signal):
    time_window = FS * 10

    signal = signal[:time_window]
    downsampled = signal[::2]
    smoothed = np.zeros_like(downsampled)

    for channel in range(3):
        smoothed[:, channel] = scipy_signal.savgol_filter(
            downsampled[:, channel],
            window_length=401,
            polyorder=4
        )
    
    envelopes = np.zeros_like(smoothed)
    for channel in range(3):
        envelopes[:, channel] = extract_envelope(smoothed[:, channel])

    combined_features = np.concatenate([smoothed, envelopes], axis=1)
    target_length = 11000
    if combined_features.shape[0] < target_length:
        combined_features = np.pad(
            combined_features,
            ((0, target_length - combined_features.shape[0]), (0, 0)),
            mode='constant'
        )
    
    return np.expand_dims(combined_features, 0)

def model(raw_signal):
    return MODEL.predict(raw_signal).squeeze() * 100

st.title("Hand Gesture Classification using EMG")

uploaded_file = st.file_uploader("Upload a signal file (CSV or TXT) @ 2000Hz", type=["csv", "txt"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=None, skiprows=3)
    raw_signal = df.iloc[:, :3].values
    processed_signal = preprocess(raw_signal)

    pred_class_probs = model(processed_signal)

    S = ''
    for class_, prob in zip(CLASS_LABELS, pred_class_probs):
        S += f"{class_}: {prob:.2f}%\n"

    st.success(S)
    st.success(f"Predicted Action: {CLASS_LABELS[pred_class_probs.argmax()]}")

    st.line_chart(np.squeeze(processed_signal))
