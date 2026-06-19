import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Data buatan (kadar gula darah vs hasil diabetes)
# 0 = Sehat, 1 = Diabetes
X = np.array([[80], [95], [105], [130], [150], [180], [200], [250]])
y = np.array([0, 0, 0, 1, 1, 1, 1, 1]) 

# Siapkan alat ukur (scaler) dan latih datanya
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Siapkan otak AI-nya (model) dan latih
model = LogisticRegression()
model.fit(X_scaled, y)

# Simpan jadi file .pkl
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("Mantap! model.pkl dan scaler.pkl berhasil dibuat.")