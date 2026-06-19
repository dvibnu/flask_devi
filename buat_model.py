import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

print("Lagi download dan baca data dataset asli...")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# Karena file CSV aslinya nggak punya baris judul, kita kasih judul sendiri biar AI-nya nggak bingung
kolom = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, header=None, names=kolom)

# Pisahin data soal (X) dan kunci jawaban (y)
# X itu 8 kriteria kesehatan, y itu hasil akhirnya (0 = sehat, 1 = diabetes)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

print("Lagi ngukur dan ngelatih AI-nya...")
# Siapkan alat ukur (scaler) biar perhitungannya adil
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Siapkan otak AI-nya (model) dan mulai belajar
model = LogisticRegression()
model.fit(X_scaled, y)

# Simpan otak dan alat ukurnya jadi file fisik
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("Mantap! model.pkl dan scaler.pkl versi ASLI udah berhasil dibuat!")