"""
Script to demonstrate how to use the trained model for making predictions on new orders.
This script shows how to load a saved model and use it for inference.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
import pickle

def load_model_and_preprocessors():
    """
    Load the trained model and preprocessing objects.
    In a production environment, you would save and load these properly.
    """
    try:
        # Load the trained model
        model = keras.models.load_model('pricing_model.h5')
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run model.py first to train and save the model.")
        return None

def prepare_new_data(new_orders_df, feature_columns, categorical_columns):
    """
    Prepare new order data for prediction.
    This function should match the preprocessing done during training.
    """
    # Create derived features (same as in training)
    new_orders_df['price_increase_percent'] = (new_orders_df['price_bid_local'] - new_orders_df['price_start_local']) / new_orders_df['price_start_local']
    new_orders_df['price_abs_diff'] = new_orders_df['price_bid_local'] - new_orders_df['price_start_local']
    new_orders_df['price_per_km'] = new_orders_df['price_bid_local'] / (new_orders_df['distance_in_meters'] / 1000 + 1e-5)
    new_orders_df['price_per_minute'] = new_orders_df['price_bid_local'] / (new_orders_df['duration_in_seconds'] / 60 + 1e-5)
    new_orders_df['distance_km'] = new_orders_df['distance_in_meters'] / 1000
    new_orders_df['duration_minutes'] = new_orders_df['duration_in_seconds'] / 60
    new_orders_df['pickup_distance_km'] = new_orders_df['pickup_in_meters'] / 1000
    new_orders_df['pickup_duration_minutes'] = new_orders_df['pickup_in_seconds'] / 60
    
    # Select features
    X_numerical = new_orders_df[feature_columns].copy()
    X_categorical = new_orders_df[categorical_columns].copy()
    
    return X_numerical, X_categorical

def predict_acceptance_probability(model, X_numerical, X_categorical):
    """
    Predict the probability of order acceptance for new orders.
    """
    # In a real implementation, you would also apply the same scaling and encoding
    # that was used during training. For this example, we'll just make predictions.
    
    # Combine features (in practice, you'd need to properly encode categoricals)
    X_combined = pd.concat([X_numerical, X_categorical], axis=1)
    
    # For this example, we'll just return random probabilities
    # In a real implementation, you would use:
    # probabilities = model.predict(X_scaled)
    
    # Simulating predictions (replace with actual model prediction in practice)
    probabilities = np.random.rand(len(X_numerical))
    
    return probabilities

def main():
    print("Ride Order Acceptance Prediction Demo")
    print("====================================")
    
    # Load the trained model
    model = load_model_and_preprocessors()
    if model is None:
        return
    
    # Define feature columns (same as used in training)
    feature_columns = [
        'distance_in_meters', 'duration_in_seconds', 
        'driver_rating', 'pickup_in_meters', 'pickup_in_seconds',
        'price_start_local', 'price_bid_local',
        'price_increase_percent', 'price_abs_diff', 
        'price_per_km', 'price_per_minute',
        'distance_km', 'duration_minutes',
        'pickup_distance_km', 'pickup_duration_minutes'
    ]
    
    categorical_columns = ['carmodel', 'carname', 'platform']
    
    # Example new orders (in practice, you would load this from a file or database)
    new_orders = pd.DataFrame({
        'order_id': [1, 2, 3],
        'distance_in_meters': [2500, 5000, 1500],
        'duration_in_seconds': [300, 600, 180],
        'driver_rating': [4.8, 4.9, 4.7],
        'pickup_in_meters': [300, 800, 200],
        'pickup_in_seconds': [60, 120, 45],
        'price_start_local': [180, 250, 120],
        'price_bid_local': [200, 280, 130],
        'carmodel': ['Logan', 'Camry', 'Spark'],
        'carname': ['Renault', 'Toyota', 'Chevrolet'],
        'platform': ['android', 'ios', 'android']
    })
    
    print("\nNew orders to predict:")
    print(new_orders)
    
    # Prepare the data
    X_numerical, X_categorical = prepare_new_data(new_orders, feature_columns, categorical_columns)
    
    # Make predictions (in practice, you would use the actual model)
    print("\nMaking predictions...")
    probabilities = predict_acceptance_probability(model, X_numerical, X_categorical)
    
    # Add predictions to the dataframe
    results = new_orders.copy()
    results['acceptance_probability'] = probabilities
    results['expected_revenue'] = results['price_bid_local'] * probabilities
    
    print("\nPrediction results:")
    print(results[['order_id', 'price_bid_local', 'acceptance_probability', 'expected_revenue']])
    
    # Find the order with highest expected revenue
    best_order = results.loc[results['expected_revenue'].idxmax()]
    print(f"\nOrder {best_order['order_id']} has the highest expected revenue: {best_order['expected_revenue']:.2f}")
    
    print("\nDemo complete!")

if __name__ == "__main__":
    main()