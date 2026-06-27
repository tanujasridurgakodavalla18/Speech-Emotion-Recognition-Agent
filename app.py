import gradio as gr
import joblib
import numpy as np
import librosa

# Load trained model
model = joblib.load("emotion_model.pkl")


# ----------------------------
# FEATURE EXTRACTION (STRICT 58 MATCH)
# ----------------------------
def extract_features(audio):
    y, sr = librosa.load(audio, sr=22050)

    # 40 MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    # Convert EVERYTHING to scalar (IMPORTANT FIX)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    chroma = float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
    rms = float(np.mean(librosa.feature.rms(y=y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))

    # Combine features
    features = np.concatenate([
        mfcc,
        [zcr, chroma, rms, centroid, bandwidth, rolloff]
    ])

    # FORCE EXACT SIZE = 58 (MODEL MATCH)
    if len(features) > 58:
        features = features[:58]
    elif len(features) < 58:
        features = np.pad(features, (0, 58 - len(features)))

    return features


# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict(audio):
    try:
        features = extract_features(audio)

        print("Final feature shape:", features.shape)  # DEBUG

        features = features.reshape(1, -1)

        pred = model.predict(features)

        return f"🎯 Predicted Emotion: {pred[0]}"

    except Exception as e:
        return f"❌ Error: {str(e)}"


# ----------------------------
# GRADIO UI
# ----------------------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload or Record Audio 🎤"),
    outputs="text",
    title="Speech Emotion Recognition AI Agent 🎤",
    description="Upload speech audio and AI will detect emotion automatically"
)

demo.launch()