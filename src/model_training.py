"""
Modul Pelatihan Model Prediksi Harga Rumah

Kelas dan fungsi untuk melatih model machine learning
"""

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

class ModelTrainer:
    def __init__(self, model_type='random_forest'):
        """
        Inisialisasi trainer model
        
        Args:
            model_type (str): Jenis model yang akan dilatih
        """
        self.model_type = model_type
        self.model = self._select_model()

    def _select_model(self):
        """
        Memilih model berdasarkan tipe
        
        Returns:
            Model machine learning
        """
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression(),
            'svr': SVR(kernel='rbf')
        }
        return models.get(self.model_type, RandomForestRegressor())

    def train_model(self, X_train, y_train):
        """
        Melatih model
        
        Args:
            X_train (array): Data fitur training
            y_train (array): Target training
        """
        self.model.fit(X_train, y_train)

    def save_model(self, filepath):
        """
        Menyimpan model yang telah dilatih
        
        Args:
            filepath (str): Path penyimpanan model
        """
        joblib.dump(self.model, filepath)
        print(f"Model disimpan di {filepath}")

    def load_model(self, filepath):
        """
        Memuat model yang tersimpan
        
        Args:
            filepath (str): Path model
        
        Returns:
            Model yang dimuat
        """
        return joblib.load(filepath)