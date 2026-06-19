from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load otak AI asli yang baru aja kita latih
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    # model_names ini wajib dikirim biar dropdown di HTML kamu nggak error
    return render_template('index.html', model_names=["Logistic Regression"])

@app.route('/predict', methods=['POST'])
def predict():
    # Tangkap ke-8 data yang diketik user di web
    pregnancies = float(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    blood_pressure = float(request.form['blood_pressure'])
    skin_thickness = float(request.form['skin_thickness'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetes_pedigree = float(request.form['diabetes_pedigree'])
    age = float(request.form['age'])
    
    # Kumpulin 8 data itu jadi satu baris rapi
    data_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    
    # Hitung prediksinya
    data_ukur = scaler.transform(data_input)
    hasil = model.predict(data_ukur)
    
    if hasil[0] == 1:
        teks_hasil = "Awas, ada indikasi Diabetes!"
    else:
        teks_hasil = "Aman, sepertinya tidak ada indikasi Diabetes."
        
    return render_template('index.html', prediction=teks_hasil, model_names=["Logistic Regression"])

if __name__ == "__main__":
    app.run(debug=True)