import os
import pickle  # Aktifkan kembali import pickle
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title("🏘️ Prediksi Harga Rumah Jabodetabek")
    
    # Pastikan direktori models ada
    if not os.path.exists('models'):
        os.makedirs('models')

    # Tambahkan deskripsi metode
    st.write("**METODE: LINEAR REGRESSION DAN RANDOM FOREST**")
    st.write("""
    Aplikasi ini menggunakan dua metode machine learning untuk memprediksi harga rumah:
    - **Linear Regression**: Untuk memahami hubungan linear antara fitur (luas tanah, luas bangunan, jumlah kamar) dengan harga rumah.
    - **Random Forest**: Model ensemble berbasis pohon keputusan yang lebih kompleks untuk menangkap hubungan non-linear dan menghasilkan prediksi yang lebih akurat.
    """)

    # Input parameter
    luas_tanah = st.number_input("Luas Tanah (m²)", min_value=0)
    luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=0)
    jumlah_kamar = st.number_input("Jumlah Kamar", min_value=0)

    if st.button("Prediksi Harga"):
        # Path file model
        model_path = 'models/house_price_model.pkl'
        scaler_path = 'models/scaler.pkl'
        imputer_path = 'models/imputer.pkl'

        # Jika file model belum ada, buat dummy model untuk contoh
        if not all(os.path.exists(path) for path in [model_path, scaler_path, imputer_path]):
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.impute import SimpleImputer

            # Buat model dummy
            model = LinearRegression()
            scaler = StandardScaler()
            imputer = SimpleImputer(strategy='mean')

            # Simpan model dummy
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            with open(imputer_path, 'wb') as f:
                pickle.dump(imputer, f)

        # Cek keberadaan file
        if all(os.path.exists(path) for path in [model_path, scaler_path, imputer_path]):
            try:
                # Muat model, scaler, dan imputer menggunakan pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                with open(imputer_path, 'rb') as f:
                    imputer = pickle.load(f)

                # Siapkan data input untuk prediksi
                input_data = np.array([[luas_tanah, luas_bangunan, jumlah_kamar]])
                
                # Gunakan imputer untuk menangani kemungkinan NaN
                input_data_imputed = imputer.transform(input_data)
                
                # Skala data
                input_data_scaled = scaler.transform(input_data_imputed)

                # Lakukan prediksi (untuk contoh, gunakan perkiraan sederhana)
                predicted_price = model.predict(input_data_scaled)
                st.success(f"Harga rumah diprediksi: Rp {predicted_price[0]:,.2f}")
                
                # ... (sisa kode visualisasi tetap sama)

            except Exception as e:
                st.error(f"Gagal memuat model atau melakukan prediksi: {e}")
        else:
            st.error("File model, scaler, atau imputer tidak ditemukan.")

if __name__ == "__main__":
    main()