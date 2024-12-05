import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Pastikan direktori models ada
if not os.path.exists('models'):
    os.makedirs('models')

# Baca dataset
df = pd.read_csv('data\jabodetabek_house_price.csv')  # Ganti dengan nama file dataset Anda

# Pisahkan fitur dan target
X = df[['land_size_m2', 'building_size_m2', 'bedrooms' ]]
y = df['price_in_rp']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisiasi Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Buat dan latih model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Simpan model dan scaler
joblib.dump(model, 'models/house_price_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("Model dan Scaler berhasil disimpan!")