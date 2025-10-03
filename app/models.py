import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class CropRecommenderModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.crop_label_encoder = LabelEncoder()
        self.feature_columns = ['Soil_Type', 'Soil_pH', 'Temperature', 'Humidity', 'N', 'P', 'K']
        self.model_path = 'models/crop_recommender_model.pkl'
        self.scaler_path = 'models/scaler.pkl'
        self.encoders_path = 'models/label_encoders.pkl'
        
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the dataset"""
        print("Loading dataset...")
        df = pd.read_csv(csv_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Crop types: {df['Crop_Type'].unique()}")
        
        # Encode categorical features
        print("Encoding categorical features...")
        self.label_encoders['Soil_Type'] = LabelEncoder()
        df['Soil_Type_Encoded'] = self.label_encoders['Soil_Type'].fit_transform(df['Soil_Type'])
        
        # Encode target variable
        df['Crop_Type_Encoded'] = self.crop_label_encoder.fit_transform(df['Crop_Type'])
        
        return df
    
    def train(self, csv_path):
        """Train the crop recommendation model"""
        df = self.load_and_preprocess_data(csv_path)
        
        # Prepare features and target
        X = df[['Soil_Type_Encoded', 'Soil_pH', 'Temperature', 'Humidity', 'N', 'P', 'K']]
        y = df['Crop_Type_Encoded']
        
        # Split data
        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        print("Evaluating model...")
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        
        # Feature importance
        feature_names = ['Soil_Type', 'Soil_pH', 'Temperature', 'Humidity', 'N', 'P', 'K']
        importances = self.model.feature_importances_
        print("\nFeature Importances:")
        for name, importance in zip(feature_names, importances):
            print(f"  {name}: {importance:.4f}")
        
        # Save model and encoders
        self.save_model()
        
        return accuracy
    
    def predict(self, soil_type, soil_ph, temperature, humidity, nitrogen, phosphorus, potassium):
        """Predict the best crop for given conditions"""
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        # Encode soil type
        try:
            soil_type_encoded = self.label_encoders['Soil_Type'].transform([soil_type])[0]
        except ValueError:
            raise ValueError(f"Unknown soil type: {soil_type}. Valid types: {list(self.label_encoders['Soil_Type'].classes_)}")
        
        # Prepare input features
        features = np.array([[
            soil_type_encoded,
            soil_ph,
            temperature,
            humidity,
            nitrogen,
            phosphorus,
            potassium
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get crop name
        crop_name = self.crop_label_encoder.inverse_transform([prediction])[0]
        
        # Get top 3 recommendations with probabilities
        top_indices = np.argsort(probabilities)[-3:][::-1]
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'crop': self.crop_label_encoder.inverse_transform([idx])[0],
                'probability': float(probabilities[idx])
            })
        
        return {
            'recommended_crop': crop_name,
            'confidence': float(probabilities[prediction]),
            'top_recommendations': recommendations
        }
    
    def save_model(self):
        """Save the trained model and encoders"""
        os.makedirs('models', exist_ok=True)
        
        print(f"Saving model to {self.model_path}...")
        joblib.dump(self.model, self.model_path)
        
        print(f"Saving scaler to {self.scaler_path}...")
        joblib.dump(self.scaler, self.scaler_path)
        
        print(f"Saving encoders to {self.encoders_path}...")
        encoders_data = {
            'label_encoders': self.label_encoders,
            'crop_label_encoder': self.crop_label_encoder
        }
        joblib.dump(encoders_data, self.encoders_path)
        
        print("Model saved successfully!")
    
    def load_model(self):
        """Load the trained model and encoders"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Please train the model first.")
        
        print("Loading model...")
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        
        encoders_data = joblib.load(self.encoders_path)
        self.label_encoders = encoders_data['label_encoders']
        self.crop_label_encoder = encoders_data['crop_label_encoder']
        
        print("Model loaded successfully!")
        
    def get_model_info(self):
        """Get information about the model"""
        if self.model is None:
            return None
        
        return {
            'model_type': 'Random Forest Classifier',
            'n_estimators': self.model.n_estimators,
            'available_crops': list(self.crop_label_encoder.classes_),
            'available_soil_types': list(self.label_encoders['Soil_Type'].classes_),
            'features': self.feature_columns
        }
