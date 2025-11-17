from flask import Flask, render_template, request, jsonify
from speechbrain.pretrained import EncoderClassifier
import torchaudio
import torch
import os
import json
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the emotion recognition model
def load_model():
    return EncoderClassifier.from_hparams(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        savedir="tmp_model"
    )

model = load_model()

# Emotion to suggestion mapping
EMOTION_SUGGESTIONS = {
    "happy": "You sound happy! Consider sharing your joy with others or doing something creative!",
    "sad": "It's okay to feel down. Consider talking to a friend or engaging in self-care activities.",
    "angry": "Take a deep breath. Try some relaxation techniques or physical activity to release tension.",
    "neutral": "You sound calm and composed. A great time for focused work or learning something new!",
    "fear": "It's natural to feel afraid sometimes. Try grounding techniques or talking to someone you trust.",
    "disgust": "This emotion can be challenging. Consider what's causing this feeling and address it directly.",
    "surprise": "Unexpected things can be exciting! Take a moment to process before reacting.",
    "calm": "You're in a peaceful state. It's a great time for meditation or creative activities."
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"audio_{timestamp}.wav")
    audio_file.save(file_path)

    try:
        # Process the audio file
        signal, sample_rate = torchaudio.load(file_path)
        
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            signal = resampler(signal)
            sample_rate = 16000

        # Get emotion prediction
        with torch.no_grad():
            predictions = model.classify_batch(signal.squeeze(0))
        
        # Get emotion probabilities
        probs = torch.softmax(predictions[1], dim=1).squeeze(0)
        emotion_labels = model.hparams.label_encoder.ind2lab
        emotion_probs = {emotion_labels[i]: float(probs[i]) for i in range(len(probs))}
        
        # Get dominant emotion
        dominant_emotion = emotion_labels[torch.argmax(probs).item()]
        suggestion = EMOTION_SUGGESTIONS.get(dominant_emotion, "Try recording another sample for analysis.")

        # Clean up
        os.remove(file_path)

        return jsonify({
            'success': True,
            'emotion': dominant_emotion,
            'probabilities': emotion_probs,
            'suggestion': suggestion
        })

    except Exception as e:
        # Clean up in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
