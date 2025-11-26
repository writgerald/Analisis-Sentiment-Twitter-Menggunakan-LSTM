import numpy as np
import tensorflow as tf
import pickle
import re
import emoji
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------------------------
#  1. INISIALISASI APLIKASI FLASK
# -------------------------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------------------------
#  2. LOAD SEMUA ARTEFAK MODEL (HANYA SEKALI SAAT APP JALAN)
# -------------------------------------------------------------------
print("Loading model...")
model = tf.keras.models.load_model('model_sentimen_lstm.h5')

print("Loading tokenizer...")
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

print("Loading label encoder...")
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Ambil parameter penting dari model (harus SAMA dengan saat training)
# Kita pakai max_length 120 (sesuai kode training terakhir kita)
MAX_LENGTH = 120 

print("Model, tokenizer, dan label encoder berhasil di-load.")

# -------------------------------------------------------------------
#  3. FUNGSI PREPROCESSING TEKS (HARUS SAMA PERSIS DENGAN TRAINING)
# -------------------------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = emoji.replace_emoji(text, replace='')  # Hapus emoji
    text = re.sub(r"http\S+|www\S+", "", text)  # Hapus URL
    text = re.sub(r"@\S+", "", text)  # Hapus mention
    text = re.sub(r"#\S+", "", text)  # Hapus hashtag
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # Hapus angka dan tanda baca
    text = text.lower()  # Lowercase
    text = re.sub(r"\s+", " ", text).strip()  # Hapus spasi berlebih
    return text

# -------------------------------------------------------------------
#  4. FUNGSI INTI UNTUK PREDIKSI
# -------------------------------------------------------------------
def predict_sentiment(text):
    """
    Fungsi lengkap untuk mengambil teks mentah, membersihkan, 
    tokenisasi, padding, prediksi, dan mengembalikan label sentimen.
    """
    # 1. Bersihkan teks
    cleaned_text = clean_text(text)
    
    # 2. Tokenisasi (ubah teks jadi angka)
    # Penting: [cleaned_text] -> ubah jadi list
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    
    # 3. Padding (samakan panjang)
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post', truncating='post')
    
    # 4. Prediksi
    prediction_probs = model.predict(padded_sequence)
    
    # 5. Ambil kelas dengan probabilitas tertinggi
    predicted_class_index = np.argmax(prediction_probs, axis=-1)[0]
    
    # 6. Ubah indeks (angka) kembali jadi label (teks)
    predicted_label = le.inverse_transform([predicted_class_index])[0]
    
    # 7. Ambil skor probabilitasnya
    confidence_score = np.max(prediction_probs)
    
    return predicted_label, confidence_score

# -------------------------------------------------------------------
#  5. ROUTING / ENDPOINT WEB
# -------------------------------------------------------------------

# Endpoint untuk halaman utama (form input)
@app.route('/')
def home():
    # Tampilkan file 'index.html' dari folder 'templates'
    return render_template('index.html')

# Endpoint untuk memproses data dari form
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 1. Ambil teks dari form input
        input_text = request.form['text']
        
        # 2. Lakukan prediksi
        label, score = predict_sentiment(input_text)
        
        # 3. Tampilkan hasilnya di halaman 'result.html'
        return render_template('result.html', 
                               text=input_text, 
                               prediction=str(label), 
                               confidence=f"{score*100:.2f}%")

# [Opsional] Endpoint untuk API (jika ingin diakses program lain)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    input_text = data['text']
    label, score = predict_sentiment(input_text)
    return jsonify({
        'input_text': input_text,
        'predicted_label': str(label),
        'confidence_score': float(score)
    })


# -------------------------------------------------------------------
#  6. JALANKAN APLIKASI
# -------------------------------------------------------------------
if __name__ == '__main__':
    # Setting host='0.0.0.0' agar bisa diakses dari luar
    app.run(host='0.0.0.0', port=5000, debug=True)