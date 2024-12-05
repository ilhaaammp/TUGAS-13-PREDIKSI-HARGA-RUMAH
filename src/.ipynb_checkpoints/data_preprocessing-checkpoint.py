"""
Modul Preprocessing Data Harga Rumah Jabodetabek

Fungsi-fungsi:
- load_data(): Memuat dataset
- handle_missing_values(): Menangani missing values
- feature_engineering(): Rekayasa fitur
- split_data(): Memisahkan data training dan testing
- scale_data(): Penskalaan fitur
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, filepath):
        """
        Inisialisasi preprocessing data
        
        Args:
            filepath (str): Path file dataset
        """
        self.filepath = filepath
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()

    def load_data(self):
        """
        Memuat dataset dari file CSV
        
        Returns:
            pandas.DataFrame: Dataset yang dimuat
        """
        try:
            self.data = pd.read_csv(self.filepath)
            print("Dataset berhasil dimuat")
            return self.data
        except Exception as e:
            print(f"Error memuat dataset: {e}")
            return None

    def handle_missing_values(self, strategy='mean'):
        """
        Menangani missing values
        
        Args:
            strategy (str): Strategi penanganan (mean/median/mode)
        """
        if strategy == 'mean':
            self.data = self.data.fillna(self.data.mean())
        elif strategy == 'median':
            self.data = self.data.fillna(self.data.median())
        elif strategy == 'mode':
            self.data = self.data.fillna(self.data.mode().iloc[0])

    def feature_engineering(self):
        """
        Rekayasa fitur tambahan
        """
        # Contoh: Menambahkan fitur baru
        self.data['price_per_sqm'] = self.data['price'] / self.data['luas_bangunan']

    def prepare_data(self, target_column='price'):
        """
        Mempersiapkan data untuk pemodelan
        
        Args:
            target_column (str): Nama kolom target
        """
        self.X = self.data.drop(target_column, axis=1)
        self.y = self.data[target_column]

    def split_data(self, test_size=0.2, random_state=42):
        """
        Memisahkan data training dan testing
        
        Args:
            test_size (float): Proporsi data testing
            random_state (int): Seed untuk reproduksibilitas
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def scale_data(self):
        """
        Penskalaan fitur numerik
        
        Returns:
            tuple: Data training dan testing yang telah diskala
        """
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        return X_train_scaled, X_test_scaled