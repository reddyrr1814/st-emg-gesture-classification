import streamlit as st
import numpy as np
import pandas as pd
import numpy as np
from scipy import signal as scipy_signal
from scipy import ndimage
import bottleneck as bn
import onnxruntime as ort

CLASS_LABELS = [
    "Come", "Stop", "Hurry up", "Crouch", "Rally Point"
]

MODEL = ort.InferenceSession("model.onnx")

INTERVALS = [(0, 11)]

def preprocess(df):
    df["time"] = pd.to_datetime(df.index.astype('float32') / 2000, unit='s')
    df.set_index("time", inplace=True)

    signals = []

    for i, (s, e) in enumerate(INTERVALS):
        s = pd.to_datetime(s, unit='s')
        e = pd.to_datetime(e, unit='s')
        segment = df[s:e]
        segment_values = segment.values

        segment = scipy_signal.resample(segment_values, 10000, axis=0)
        smoothed = scipy_signal.savgol_filter(
            segment,
            window_length=501,
            polyorder=2,
            axis=0
        )

        stds = np.std(smoothed, axis=0, keepdims=True)
        norm_smoothed = (smoothed / stds)

        dilated = np.vstack([
            ndimage.grey_dilation(norm_smoothed[:, 0], size=20),
            ndimage.grey_dilation(norm_smoothed[:, 1], size=20),
            ndimage.grey_dilation(norm_smoothed[:, 2], size=20)
        ])

        eroded = np.vstack([
            ndimage.grey_erosion(norm_smoothed[:, 0], size=20),
            ndimage.grey_erosion(norm_smoothed[:, 1], size=20),
            ndimage.grey_erosion(norm_smoothed[:, 2], size=20)
        ])

        smooth_dilated = bn.move_mean(dilated, 10, axis=1)
        smooth_eroded = bn.move_mean(eroded, 10, axis=1)

        features = np.vstack([
            segment.T,
            norm_smoothed.T,
            smooth_dilated,
            smooth_eroded
        ])

        features = np.nan_to_num(features)
        signals.append(features)

    return signals

st.title("Hand Gesture Classification using EMG")

uploaded_file = st.file_uploader("Upload a signal file (CSV or TXT) @ 2000Hz", type=["csv", "txt"])
value = st.number_input("Start time duration", min_value=0, step=1)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    INTERVALS = [(value, value + 11)]

    processed_signal = np.asarray(preprocess(df))
    print(processed_signal.shape)


    pred_class_probs = MODEL.run(None, {"input": np.swapaxes(processed_signal, 1, 2).astype('float32')})[0][0]
    pred_class_probs = np.exp(pred_class_probs)
    pred_class_probs /= np.sum(pred_class_probs)

    S = ''
    for class_, prob in zip(CLASS_LABELS, pred_class_probs):
        S += f"{class_}: {float(prob) * 100:.4f}%\n"

    st.success(S)
    st.success(f"Predicted Action: {CLASS_LABELS[pred_class_probs.argmax()]}")

    short_signal = scipy_signal.resample(processed_signal, 2000, axis=-1)
    st.line_chart(np.squeeze(short_signal.T))
