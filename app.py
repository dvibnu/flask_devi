from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load 2 otak yang tersisa
model_svc = pickle.load(open('model_svc.pkl', 'rb'))
model_dt = pickle.load(open('model_dt.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Daftar nama ini yang bakal muncul di kotak pilihan web kamu (cuma 2 sekarang)
daftar_model = ["SVC", "Decision Tree"]

@app.route('/')
def home():
    return render_template('index.html', model_names=daftar_model)

@app.route('/predict', methods=['POST'])
def predict():
    pilihan_model = request.form['model']
    
    pregnancies = float(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    blood_pressure = float(request.form['blood_pressure'])
    skin_thickness = float(request.form['skin_thickness'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetes_pedigree = float(request.form['diabetes_pedigree'])
    age = float(request.form['age'])
    
    data_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    data_ukur = scaler.transform(data_input)
    
    # KONDISI: Kalau yang dipilih SVC, pakai otak SVC. Kalau bukan, otomatis pakai Decision Tree.
    if pilihan_model == "SVC":
        otak_yg_dipakai = model_svc
    else:
        otak_yg_dipakai = model_dt
        
    hasil = otak_yg_dipakai.predict(data_ukur)
    
    if hasil[0] == 1:
        teks_hasil = f"Awas, ada indikasi Diabetes! (Dihitung pakai: {pilihan_model})"
    else:
        teks_hasil = f"Aman, tidak ada indikasi Diabetes. (Dihitung pakai: {pilihan_model})"
        
    return render_template('index.html', prediction=teks_hasil, model_names=daftar_model)

if __name__ == "__main__":
    app.run(debug=True)