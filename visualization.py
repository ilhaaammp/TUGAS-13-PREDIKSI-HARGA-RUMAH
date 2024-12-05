import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go

def load_data():
    """Memuat dataset"""
    df = pd.read_csv('dataset_rumah.csv')
    return df

def show_dataset_info(df):
    """Menampilkan informasi dataset"""
    st.subheader("📊 Informasi Dataset")
    
    # Statistik Deskriptif
    st.write("### Statistik Deskriptif")
    st.dataframe(df.describe())
    
    # Informasi Kolom
    st.write("### Informasi Kolom")
    col_info = pd.DataFrame({
        'Nama Kolom': df.columns,
        'Tipe Data': df.dtypes,
        'Jumlah Non-Null': df.notna().sum(),
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(col_info)

def create_visualizations(df):
    """Membuat berbagai visualisasi"""
    st.subheader("📈 Visualisasi Data")
    
    # Pilihan Visualisasi
    viz_option = st.selectbox("Pilih Jenis Visualisasi", [
        "Distribusi Harga Rumah",
        "Korelasi Antar Fitur",
        "Scatter Plot Luas Tanah vs Harga",
        "Box Plot Jumlah Kamar",
        "Histogram Luas Bangunan"
    ])
    
    # Matplotlib & Seaborn Visualizations
    plt.figure(figsize=(10, 6))
    
    if viz_option == "Distribusi Harga Rumah":
        plt.title("Distribusi Harga Rumah")
        sns.histplot(df['harga'], kde=True)
        st.pyplot(plt)
    
    elif viz_option == "Korelasi Antar Fitur":
        plt.title("Korelasi Antar Fitur")
        korelasi = df.corr()
        sns.heatmap(korelasi, annot=True, cmap='coolwarm')
        st.pyplot(plt)
    
    elif viz_option == "Scatter Plot Luas Tanah vs Harga":
        plt.title("Luas Tanah vs Harga Rumah")
        plt.scatter(df['luas_tanah'], df['harga'])
        plt.xlabel("Luas Tanah")
        plt.ylabel("Harga")
        st.pyplot(plt)
    
    elif viz_option == "Box Plot Jumlah Kamar":
        plt.title("Distribusi Harga Berdasarkan Jumlah Kamar")
        sns.boxplot(x='jumlah_kamar', y='harga', data=df)
        st.pyplot(plt)
    
    elif viz_option == "Histogram Luas Bangunan":
        plt.title("Histogram Luas Bangunan")
        plt.hist(df['luas_bangunan'], bins=20)
        plt.xlabel("Luas Bangunan")
        plt.ylabel("Frekuensi")
        st.pyplot(plt)
    
    # Plotly Interactive Visualization
    st.subheader("🔍 Visualisasi Interaktif")
    
    # Scatter 3D Interaktif
    fig = px.scatter_3d(df, x='luas_tanah', y='luas_bangunan', z='harga', 
                        color='jumlah_kamar', 
                        title='Hubungan 3D Luas Tanah, Bangunan, dan Harga')
    st.plotly_chart(fig)

def main_visualization():
    st.title("🏡 Visualisasi Dataset Rumah")
    
    # Muat Dataset
    df = load_data()
    
    # Tab Visualisasi
    tab1, tab2, tab3 = st.tabs([
        "Informasi Dataset", 
        "Visualisasi Statis", 
        "Data Mentah"
    ])
    
    with tab1:
        show_dataset_info(df)
    
    with tab2:
        create_visualizations(df)
    
    with tab3:
        st.subheader("📋 Data Mentah")
        st.dataframe(df)

if __name__ == "__main__":
    main_visualization()