from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load otak AI yang tadi kita bikin
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil angka gula darah yang diketik di web
    gula_darah = float(request.form['gula'])
    
    # Hitung prediksinya
    data_ukur = scaler.transform([[gula_darah]])
    hasil = model.predict(data_ukur)
    
    if hasil[0] == 1:
        teks_hasil = "Awas, ada indikasi Diabetes!"
    else:
        teks_hasil = "Aman, sepertinya tidak Diabetes."
        
    return render_template('index.html', hasil_prediksi=teks_hasil)

if __name__ == "__main__":
    app.run(debug=True)