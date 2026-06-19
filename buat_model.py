import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

print("Lagi baca data diabetes.csv...")
df = pd.read_csv('diabetes.csv')

# Pisahin soal dan kunci jawaban
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Alat ukur biar adil
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Lagi ngelatih 2 OTAK! Tunggu bentar...")

# 1. Bikin otak SVC (Support Vector Classifier)
model_svc = SVC()
model_svc.fit(X_scaled, y)

# 2. Bikin otak Decision Tree
model_dt = DecisionTreeClassifier()
model_dt.fit(X_scaled, y)

# Simpan alat ukur dan kedua otak AI-nya
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(model_svc, open('model_svc.pkl', 'wb'))
pickle.dump(model_dt, open('model_dt.pkl', 'wb'))

print("Berhasil! Otak SVC dan Decision Tree udah siap dipakai!")