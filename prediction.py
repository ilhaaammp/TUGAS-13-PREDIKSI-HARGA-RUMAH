import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_model():
    """Load the trained model"""
    model_path = 'models/house_price_model.pkl'
    scaler_path = 'models/scaler.pkl'
    imputer_path = 'models/imputer.pkl'

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)

    return model, scaler, imputer

def make_prediction(input_data):
    """Make a prediction using the input data"""
    model, scaler, imputer = load_model()

    # Impute missing values
    input_data_imputed = imputer.transform(input_data)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data_imputed)

    # Make the prediction
    prediction = model.predict(input_data_scaled)

    return prediction

def main():
    """Main function for prediction"""
    # Example input data
    input_data = pd.DataFrame({'luas_tanah': [100], 'luas_bangunan': [50], 'jumlah_kamar': [3]})

    prediction = make_prediction(input_data)

    print(f"Predicted house price: {prediction[0]:.2f}")

if __name__ == "__main__":
    main()