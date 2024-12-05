import os
import sys

# Impor library dengan penanganan yang lebih aman
try:
    import pickle
    import numpy as np
    import streamlit as st
    import pandas as pd
    
    # Tambahkan backend non-interaktif untuk matplotlib
    import matplotlib
    matplotlib.use('Agg')
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
except ImportError as e:
    st.error(f"Error importing module: {e}")
    st.error("Pastikan semua library sudah terinstal.")
    st.stop()

def main():
    st.title("🏘️ Prediksi Harga Rumah Jabodetabek")
    
    # Pastikan direktori models ada
    if not os.path.exists('models'):
        os.makedirs('models')

    # Tambahkan deskripsi metode
    st.write("**METODE: PREDIKSI HARGA RUMAH**")
    st.write("""
    Aplikasi ini menggunakan metode sederhana untuk memprediksi harga rumah 
    berdasarkan luas tanah, luas bangunan, dan jumlah kamar.
    """)

    # Input parameter
    luas_tanah = st.number_input("Luas Tanah (m²)", min_value=0, value=100)
    luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=0, value=100)
    jumlah_kamar = st.number_input("Jumlah Kamar", min_value=0, value=2)

    def prediksi_harga_sederhana(luas_tanah, luas_bangunan, jumlah_kamar):
        """
        Fungsi prediksi harga sederhana
        """
        # Faktor konstanta
        harga_per_m2_tanah = 5_000_000  # Rp 5 juta per m2
        harga_per_m2_bangunan = 10_000_000  # Rp 10 juta per m2
        harga_per_kamar = 50_000_000  # Rp 50 juta per kamar

        # Hitung prediksi
        prediksi = (
            (luas_tanah * harga_per_m2_tanah) + 
            (luas_bangunan * harga_per_m2_bangunan) + 
            (jumlah_kamar * harga_per_kamar)
        )
        
        return prediksi

    if st.button("Prediksi Harga"):
        try:
            # Lakukan prediksi
            predicted_price = prediksi_harga_sederhana(
                luas_tanah, 
                luas_bangunan, 
                jumlah_kamar
            )
            
            # Tampilkan hasil prediksi
            st.success(f"Estimasi Harga Rumah: Rp {predicted_price:,.2f}")
            
            # Visualisasi kontribusi
            st.subheader("📊 Kontribusi Faktor Harga")
            
            # Hitung persentase kontribusi
            total = predicted_price
            kontribusi = [
                (luas_tanah * 5_000_000 / total) * 100,
                (luas_bangunan * 10_000_000 / total) * 100,
                (jumlah_kamar * 50_000_000 / total) * 100
            ]
            
            # Buat plot
            plt.figure(figsize=(10, 6))
            plt.bar(
                ['Luas Tanah', 'Luas Bangunan', 'Jumlah Kamar'], 
                kontribusi, 
                color=['blue', 'green', 'red']
            )
            plt.title('Kontribusi Faktor terhadap Harga Rumah')
            plt.xlabel('Faktor')
            plt.ylabel('Kontribusi (%)')
            plt.ylim(0, 100)
            
            # Tambahkan label persentase
            for i, v in enumerate(kontribusi):
                plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
            
            st.pyplot(plt)
            plt.close()

            # Rincian detail
            st.subheader("📝 Rincian Perhitungan")
            st.write(f"**Luas Tanah:** {luas_tanah} m² (Rp {luas_tanah * 5_000_000:,})")
            st.write(f"**Luas Bangunan:** {luas_bangunan} m² (Rp {luas_bangunan * 10_000_000:,})")
            st.write(f"**Jumlah Kamar:** {jumlah_kamar} (Rp {jumlah_kamar * 50_000_000:,})")

        except Exception as e:
            st.error(f"Terjadi kesalahan dalam prediksi: {e}")

if __name__ == "__main__":
    main()