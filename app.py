from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from flask_cors import CORS
import pickle

emotion_dict = {
    0: 'Angry',
    1: 'Calm',
    2: 'Disgust',
    3: 'Fearful',
    4: 'Happy',
    5: 'Neutral',
    6: 'Sad',
    7: 'Surprised'
}


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}}, methods=["GET", "POST", "OPTIONS"])


# Load your trained modelp

# Model path
MODEL_PATH = os.path.join("MODEL", "final_improved_model.h5")
MODEL = load_model(MODEL_PATH)

# Label encoder path
DATAFILE_PATH = os.path.join("DATAFILE", "label_encoder.pkl")
with open(DATAFILE_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# Label mapping (update if needed)
CLASS_LABELS = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad']

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # Chroma
    stft = np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    # RMS Energy
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms.T, axis=0)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr.T, axis=0)

    # Combine all features
    combined_features = np.hstack([mfccs_mean, chroma_mean, rms_mean, zcr_mean])
    return np.expand_dims(combined_features, axis=0)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['file']
    file_path = "./uploaded_audio.wav"
    file.save(file_path)

    try:
        from pydub import AudioSegment
        sound = AudioSegment.from_file(file_path)
        sound.export(file_path, format="wav")

        input_data = extract_features(file_path)
        if input_data is None:
            return jsonify({'error': 'Failed to extract features'}), 500


        prediction = model.predict(input_data)[0]
        predicted_class = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        confidence = float(np.max(prediction))

        return jsonify({
            'emotion': predicted_label.capitalize(),
            'confidence': round(confidence, 3)
            
        })

        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
