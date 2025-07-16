#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import scipy.signal
import soundfile as sf
import os
import io
import tempfile

# Force CPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""
tf.config.set_visible_devices([], 'GPU')

def detect_speech(y, sr):
    frame_length = int(0.03 * sr)  # 30ms frames
    hop_length = frame_length // 2
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    threshold = np.mean(energy) + 0.5 * np.std(energy)
    speech_frames = energy > threshold
    speech_frames = scipy.signal.medfilt(speech_frames.astype(int), kernel_size=5)
    st.write(f"Speech frames shape: {speech_frames.shape}")
    return speech_frames

def extract_mfcc(y, sr, frame_length=512, hop_length=256):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length, n_fft=frame_length)
    st.write(f"MFCC shape: {mfcc.T.shape}")
    return mfcc.T

def create_sequences(features, labels, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(labels[i + seq_length - 1])
    return np.array(X), np.array(y)

def build_model(seq_length, n_mfcc, num_speakers):
    model = models.Sequential([
        layers.Input(shape=(seq_length, n_mfcc)),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.LSTM(128, return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_speakers, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def estimate_num_speakers(mfcc, max_speakers=10):
    best_score = -1
    best_k = 2
    for k in range(2, max_speakers + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(mfcc)
        score = silhouette_score(mfcc, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

def get_speaker_segments(labels, times, num_speakers):
    segments = {i: [] for i in range(num_speakers)}
    for speaker in range(num_speakers):
        mask = labels == speaker
        start_time = None
        for i, is_speaker in enumerate(mask):
            if is_speaker and start_time is None:
                start_time = times[i]
            elif not is_speaker and start_time is not None:
                segments[speaker].append((start_time, times[i]))
                start_time = None
        if start_time is not None and i < len(times) - 1:
            segments[speaker].append((start_time, times[i]))
    return segments

def save_audio_segment(y, sr, start_time, end_time, filename):
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    segment = y[start_sample:end_sample]
    with io.BytesIO() as buffer:
        sf.write(buffer, segment, sr, format='WAV')
        return buffer.getvalue()

def compute_der(true_labels, pred_labels):
    if len(true_labels) != len(pred_labels):
        return None
    errors = np.sum(true_labels != pred_labels)
    return (errors / len(true_labels)) * 100

def generate_manual_labels(num_frames, segment_ranges, num_speakers):
    labels = np.zeros(num_frames, dtype=int)
    for (start, end), speaker in segment_ranges:
        start_idx = int(start * num_frames / (num_frames * 0.016))  # Approx frame duration
        end_idx = int(end * num_frames / (num_frames * 0.016))
        labels[start_idx:end_idx] = speaker
    return labels

def process_diarization(audio_file, num_speakers, manual_segment_ranges, sr=16000):
    try:
        # Load audio
        y, sr = librosa.load(audio_file, sr=sr, mono=True)
        st.write(f"Loaded audio: {len(y)/sr:.2f}s, {sr} Hz")
        audio_duration = len(y) / sr
        
        # Speech Detection
        speech_frames = detect_speech(y, sr)
        frame_duration = 0.015  # 15ms per frame
        speech_timestamps = []
        start_time = None
        for i, is_speech in enumerate(speech_frames):
            if is_speech and start_time is None:
                start_time = i * frame_duration
            elif not is_speech and start_time is not None:
                speech_timestamps.append((start_time, i * frame_duration))
                start_time = None
        if start_time is not None:
            speech_timestamps.append((start_time, len(speech_frames) * frame_duration))
        
        # Extract MFCC
        frame_length = 512
        hop_length = 256
        mfcc = extract_mfcc(y, sr, frame_length, hop_length)
        num_frames = mfcc.shape[0]
        st.write(f"Number of MFCC frames: {num_frames}")
        
        # Estimate number of speakers
        estimated_speakers = estimate_num_speakers(mfcc, max_speakers=10)
        st.write(f"**Estimated number of speakers:** {estimated_speakers}")
        
        # Use manual labels if provided, else simulate
        if manual_segment_ranges:
            labels = generate_manual_labels(num_frames, manual_segment_ranges, num_speakers)
        else:
            labels = np.array([(i // 50) % num_speakers for i in range(num_frames)])
        num_speakers = len(np.unique(labels))
        st.write(f"Number of speakers used: {num_speakers}")
        
        # Prepare sequences
        seq_length = 20
        X, y_labels = create_sequences(mfcc, labels, seq_length)
        st.write(f"X shape: {X.shape}, y_labels shape: {y_labels.shape}")
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_labels)
        y_encoded = tf.keras.utils.to_categorical(y_encoded)
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        model = build_model(seq_length, mfcc.shape[1], num_speakers)
        history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test), verbose=1)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        predicted_speakers = le.inverse_transform(y_pred_labels)
        st.write(f"Predicted speakers shape: {predicted_speakers.shape}")
        
        # K-means baseline
        kmeans = KMeans(n_clusters=num_speakers, random_state=42)
        kmeans_labels = kmeans.fit_predict(mfcc)
        kmeans_pred_labels = kmeans_labels[-len(predicted_speakers):]
        
        # Silhouette Score
        silhouette_cnn_lstm = silhouette_score(X_test.reshape(X_test.shape[0], -1), y_pred_labels)
        silhouette_kmeans = silhouette_score(mfcc[-len(kmeans_pred_labels):], kmeans_pred_labels)
        st.write(f"**Silhouette Score (CNN-LSTM):** {silhouette_cnn_lstm:.4f}")
        st.write(f"**Silhouette Score (K-means):** {silhouette_kmeans:.4f}")
        
        # DER (if manual labels provided)
        der = compute_der(y_labels[-len(predicted_speakers):], predicted_speakers) if manual_segment_ranges else None
        if der is not None:
            st.write(f"**Diarization Error Rate (DER):** {der:.2f}%")
        
        # Detect overlaps
        times = np.linspace(0, len(y) / sr, num_frames)
        speaker_changes = np.diff(labels)
        overlap_times = times[:-1][np.abs(speaker_changes) > 0]
        
        # Get speaker segments
        speaker_segments = get_speaker_segments(labels, times, num_speakers)
        
        # Visualize
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
        
        # Waveform with speaker labels
        audio_times = np.linspace(0, len(y) / sr, len(y))
        ax1.plot(audio_times, y, label='Audio Waveform', alpha=0.3)
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'gray'][:num_speakers]
        for speaker in range(num_speakers):
            mask = labels == speaker
            frame_times = times[mask]
            sample_indices = (frame_times * sr).astype(int)
            sample_indices = sample_indices[sample_indices < len(y)]
            ax1.plot(audio_times[sample_indices], y[sample_indices], label=f'Speaker {speaker}', color=colors[speaker], alpha=0.8)
        ax1.set_title("Waveform with Speaker Labels")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.legend()
        
        # Speech detection
        speech_hop = int(0.015 * sr)
        speech_mask = np.repeat(speech_frames, speech_hop)
        speech_mask = speech_mask[:len(y)]
        ax2.plot(audio_times, y * speech_mask, label='Speech Segments', color='purple')
        ax2.set_title("Speech Detection")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.legend()
        
        # Diarization output
        ax3.plot(times, labels, label='Speaker Labels', color='black')
        for t in overlap_times:
            ax3.axvline(t, color='red', linestyle='--', alpha=0.5)
        ax3.set_title("Diarization Output with Overlaps (red lines)")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Speaker ID")
        ax3.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display timestamps
        st.write("**Speech Segments (start, end):**")
        for start, end in speech_timestamps:
            st.write(f"{start:.2f}s - {end:.2f}s")
        
        # Display and download speaker segments
        st.write("**Speaker Segments:**")
        for speaker in range(num_speakers):
            st.write(f"**Speaker {speaker} Segments:**")
            for i, (start, end) in enumerate(speaker_segments[speaker]):
                st.write(f"Segment {i+1}: {start:.2f}s - {end:.2f}s")
                audio_buffer = save_audio_segment(y, sr, start, end, f"speaker_{speaker}_segment_{i+1}.wav")
                st.download_button(
                    label=f"Download Speaker {speaker} Segment {i+1} ({start:.2f}s - {end:.2f}s)",
                    data=audio_buffer,
                    file_name=f"speaker_{speaker}_segment_{i+1}.wav",
                    mime="audio/wav",
                    key=f"speaker_{speaker}_segment_{i+1}"
                )
        
        # Display and download overlap segments
        st.write("**Overlap Times:**")
        for i, t in enumerate(overlap_times):
            start = max(0, t - 0.5)
            end = min(len(y) / sr, t + 0.5)
            st.write(f"Overlap {i+1}: {t:.2f}s")
            audio_buffer = save_audio_segment(y, sr, start, end, f"overlap_{i+1}.wav")
            st.download_button(
                label=f"Download Overlap {i+1} ({start:.2f}s - {end:.2f}s)",
                data=audio_buffer,
                file_name=f"overlap_{i+1}.wav",
                mime="audio/wav",
                key=f"overlap_{i+1}"
            )
        
        # Save model
        model.save('speaker_diarization_model.keras')
        st.success("Diarization complete! Model saved as 'speaker_diarization_model.keras'")
        
        return True
    
    except Exception as e:
        st.error(f"Processing failed: {e}")
        return False

# Streamlit UI
st.title("Speaker Diarization Web App")
st.write("Upload a .wav file (16kHz, mono preferred) to perform speaker diarization.")

# Number of speakers slider
num_speakers = st.slider("Select number of speakers", min_value=1, max_value=10, value=2, key="num_speakers")

# Manual labeling with sliders
st.write("**Manual Labeling (Optional)**")
st.write("Use sliders to label speaker segments. Add segments as needed. Leave empty to use simulated labels.")
num_segments = st.number_input("Number of segments to label", min_value=0, max_value=20, value=0, step=1, key="num_segments")
manual_segment_ranges = []
if num_segments > 0:
    for i in range(num_segments):
        st.write(f"**Segment {i+1}**")
        start_time = st.slider(f"Start time (s)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key=f"start_{i}")
        end_time = st.slider(f"End time (s)", min_value=start_time, max_value=100.0, value=start_time+1.0, step=0.1, key=f"end_{i}")
        speaker_id = st.slider(f"Speaker ID", min_value=0, max_value=num_speakers-1, value=0, step=1, key=f"speaker_{i}")
        manual_segment_ranges.append(((start_time, end_time), speaker_id))

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"], key="audio_uploader")
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    
    st.write(f"Uploaded file: {uploaded_file.name}")
    if st.button("Process Diarization", key="process_button"):
        with st.spinner("Processing audio..."):
            success = process_diarization(tmp_file_path, num_speakers, manual_segment_ranges)
        if success:
            st.write("Processing complete! See results below.")
    
    # Clean up temp file
    os.unlink(tmp_file_path)

