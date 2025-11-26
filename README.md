# Analisis Sentimen Twitter Menggunakan LSTM

Proyek ini bertujuan untuk melakukan Analisis Sentimen pada data media sosial Twitter (sekarang X) terkait topik PPKM (Pemberlakuan Pembatasan Kegiatan Masyarakat) yang diperpanjang. Model dibangun menggunakan metode Deep Learning, khususnya Long Short-Term Memory (LSTM), untuk mengklasifikasikan sentimen tweet menjadi positif, negatif, atau netral.

Selain model Jupyter Notebook, repositori ini juga menyertakan folder ProyekWebSentimen yang kemungkinan berisi implementasi web (Web App) untuk mendemonstrasikan model secara interaktif.

## 📂 Struktur Repositori
* Project_Deep_Learning_Terakhir_FIX_NOREVISI(12).ipynb: File Jupyter Notebook utama yang berisi seluruh tahapan pengembangan model, mulai dari:

* Pemuatan data (Data Loading).

* Pra-pemrosesan teks (Cleaning, Case Folding, Tokenization, Stopword Removal, Stemming).

* Pembangunan model LSTM (Embedding Layer, LSTM Layer, Dense Layer).

Pelatihan dan Evaluasi Model (Training & Evaluation).

ProyekWebSentimen/: Direktori yang berisi source code untuk aplikasi web (kemungkinan berbasis Django atau Flask) untuk deployment model analisis sentimen.

🛠️ Teknologi yang Digunakan
Bahasa Pemrograman: Python

Machine Learning / Deep Learning: TensorFlow, Keras

Data Processing: Pandas, NumPy

NLP Tools: NLTK, Sastrawi (untuk stemming bahasa Indonesia), Scikit-learn

Web Framework: (Perlu dicek di dalam folder ProyekWebSentimen, kemungkinan Django atau Flask)

🚀 Cara Menjalankan
Prasyarat
Pastikan Anda telah menginstal Python dan library yang diperlukan. Anda dapat menginstalnya menggunakan pip:

Bash

pip install tensorflow pandas numpy nltk sastrawi scikit-learn matplotlib
1. Menjalankan Notebook (Model Training)
Untuk melihat proses analisis data dan pelatihan model:

Clone repositori ini:

Bash

git clone https://github.com/writgerald/Analisis-Sentiment-Twitter-Menggunakan-LSTM.git
Buka Jupyter Notebook atau Google Colab.

Jalankan file Project_Deep_Learning_Terakhir_FIX_NOREVISI(12).ipynb.

2. Menjalankan Aplikasi Web
Untuk menjalankan antarmuka web yang ada di folder ProyekWebSentimen:

Masuk ke direktori web:

Bash

cd Analisis-Sentiment-Twitter-Menggunakan-LSTM/ProyekWebSentimen
Install dependencies (jika ada file requirements.txt):

Bash

pip install -r requirements.txt
Jalankan server (tergantung framework yang digunakan):

Jika Django:

Bash

python manage.py runserver
Jika Flask:

Bash

python app.py
Buka browser dan akses alamat localhost yang muncul (biasanya http://127.0.0.1:8000 atau http://127.0.0.1:5000).

📊 Alur Kerja (Workflow)
Crawling Data: Mengambil data tweet terkait "PPKM Diperpanjang".

Preprocessing: Membersihkan data dari simbol, angka, mengubah ke huruf kecil, dan menghapus kata-kata tidak penting (stopwords).

Tokenizing & Padding: Mengubah teks menjadi urutan angka (sequence) agar dapat diproses oleh LSTM.

Modeling: Melatih model LSTM untuk mengenali pola sentimen.

Evaluasi: Mengukur akurasi model menggunakan data testing.

🤝 Kontribusi
Kontribusi selalu diterima! Silakan buat Pull Request atau buka Issue jika Anda menemukan bug atau ingin menambahkan fitur baru.

📝 Kredit
Dibuat oleh writgerald.
