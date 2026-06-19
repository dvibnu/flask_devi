import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

print("Lagi baca data dari file diabetes.csv yang baru...")
# Langsung baca file CSV yang ada di folder yang sama
df = pd.read_csv('diabetes.csv')

# Pisahin data soal (X) dan kunci jawaban (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

print("Lagi ngukur dan ngelatih AI-nya...")
# Siapkan alat ukur (scaler) biar perhitungannya adil
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Siapkan otak AI-nya (model) dan mulai belajar
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# Simpan otak dan alat ukurnya jadi file fisik
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("Mantap! model.pkl dan scaler.pkl dari dataset BARU udah berhasil dibuat!")