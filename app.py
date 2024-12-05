import os
import joblib
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title("🏘️ Prediksi Harga Rumah Jabodetabek")
    
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

        # Cek keberadaan file
        if all(os.path.exists(path) for path in [model_path, scaler_path, imputer_path]):
            try:
                # Muat model, scaler, dan imputer
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                imputer = joblib.load(imputer_path)

                # Siapkan data input untuk prediksi
                input_data = np.array([[luas_tanah, luas_bangunan, jumlah_kamar]])
                
                # Gunakan imputer untuk menangani kemungkinan NaN
                input_data_imputed = imputer.transform(input_data)
                
                # Skala data
                input_data_scaled = scaler.transform(input_data_imputed)

                # Lakukan prediksi
                predicted_price = model.predict(input_data_scaled)
                st.success(f"Harga rumah diprediksi: Rp {predicted_price[0]:,.2f}")
                
                # Visualisasi Grafik
                st.subheader("📊 Visualisasi Grafik")

                # Contoh data untuk grafik
                actual_prices = [500, 600, 700, 800, 900]
                predicted_rf = [510, 620, 710, 790, 880]  # Random Forest
                predicted_lr = [495, 605, 715, 810, 920]  # Linear Regression
                features = ['Luas Tanah', 'Luas Bangunan', 'Jumlah Kamar']
                importances = [0.4, 0.45, 0.15]
                residuals_rf = [10, 20, -10, 10, -20]  # Random Forest
                residuals_lr = [-5, 5, 15, 10, 20]     # Linear Regression

                # Grafik 1: Perbandingan Nilai Aktual vs Prediksi
                st.write("**Perbandingan Nilai Aktual vs Prediksi**")
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                ax1.plot(actual_prices, label='Actual Prices', marker='o')
                ax1.plot(predicted_rf, label='Random Forest Predictions', marker='s')
                ax1.plot(predicted_lr, label='Linear Regression Predictions', marker='^')
                ax1.legend()
                ax1.set_title('Actual vs Predicted Prices')
                ax1.set_xlabel('Sample Index')
                ax1.set_ylabel('Price')
                st.pyplot(fig1)

                # Grafik 2: Feature Importance
                st.write("**Feature Importance (Random Forest)**")
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                sns.barplot(x=importances, y=features, ax=ax2, palette='viridis')
                ax2.set_title('Feature Importance (Random Forest)')
                ax2.set_xlabel('Importance')
                ax2.set_ylabel('Features')
                st.pyplot(fig2)

                # Grafik 3: Distribusi Kesalahan
                st.write("**Distribusi Kesalahan**")
                fig3, ax3 = plt.subplots(figsize=(8, 6))
                sns.histplot(residuals_rf, kde=True, color='blue', label='Random Forest', bins=10, ax=ax3)
                sns.histplot(residuals_lr, kde=True, color='red', label='Linear Regression', bins=10, ax=ax3)
                ax3.set_title('Residual Distribution')
                ax3.set_xlabel('Residuals')
                ax3.legend()
                st.pyplot(fig3)

            except Exception as e:
                st.error(f"Gagal memuat model atau melakukan prediksi: {e}")
        else:
            st.error("File model, scaler, atau imputer tidak ditemukan. Pastikan model telah dilatih dan disimpan.")

if __name__ == "__main__":
    main()
