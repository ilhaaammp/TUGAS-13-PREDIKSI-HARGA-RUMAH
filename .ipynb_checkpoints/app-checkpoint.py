import os
import joblib
import numpy as np
import streamlit as st
import pandas as pd
import streamlit as st

def main():
    st.title("🏘️ Prediksi Harga Rumah Jabodetabek")

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

            except Exception as e:
                st.error(f"Gagal memuat model atau melakukan prediksi: {e}")
        else:
            st.error("File model, scaler, atau imputer tidak ditemukan. Pastikan model telah dilatih dan disimpan.")

if __name__ == "__main__":
    main()
    