import tensorflow as tf
model = tf.keras.models.load_model('speaker_diarization_model.keras')
print("Model loaded successfully")