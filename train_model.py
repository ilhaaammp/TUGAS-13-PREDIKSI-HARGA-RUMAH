import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer  # Tambahkan imputer
import joblib
import os

# Pastikan direktori models ada
if not os.path.exists('models'):
    os.makedirs('models')

# Baca dataset
df = pd.read_csv('data\jabodetabek_house_price.csv')  # Sesuaikan nama file dataset Anda

# Cetak informasi dataset untuk debugging
print("Informasi Dataset:")
print(df.info())
print("\nCek Nilai NaN:")
print(df.isnull().sum())

# Pisahkan fitur dan target
X = df[['land_size_m2', 'building_size_m2', 'bedrooms']]
y = df['price_in_rp']

# Tangani nilai NaN menggunakan SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Inisiasi Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Buat dan latih model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Simpan model, scaler, dan imputer
joblib.dump(model, 'models/house_price_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(imputer, 'models/imputer.pkl')

print("Model, Scaler, dan Imputer berhasil disimpan!")

# Evaluasi model (opsional)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")