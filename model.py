import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('train.csv')

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nTarget variable distribution:")
print(df['is_done'].value_counts())

# Data preprocessing
# Convert target variable to binary (0 for cancel, 1 for done)
df['is_done'] = df['is_done'].map({'cancel': 0, 'done': 1})

# Feature engineering
# Create derived features as mentioned in the task
df['price_increase_percent'] = (df['price_bid_local'] - df['price_start_local']) / df['price_start_local']
df['price_abs_diff'] = df['price_bid_local'] - df['price_start_local']
df['price_per_km'] = df['price_bid_local'] / (df['distance_in_meters'] / 1000 + 1e-5)  # Adding small value to avoid division by zero
df['price_per_minute'] = df['price_bid_local'] / (df['duration_in_seconds'] / 60 + 1e-5)
df['distance_km'] = df['distance_in_meters'] / 1000
df['duration_minutes'] = df['duration_in_seconds'] / 60
df['pickup_distance_km'] = df['pickup_in_meters'] / 1000
df['pickup_duration_minutes'] = df['pickup_in_seconds'] / 60

# Handle missing values
print("\nMissing values:")
print(df.isnull().sum())

# Fill missing values in driver_rating with median
df['driver_rating'].fillna(df['driver_rating'].median(), inplace=True)

# Select features for the model (removed only duplicates)
feature_columns = [
    'distance_km', 'duration_minutes',  # only converted versions
    'driver_rating', 'pickup_distance_km', 'pickup_duration_minutes',
    'price_start_local', 'price_bid_local',
    'price_increase_percent', 'price_abs_diff', 
    'price_per_km', 'price_per_minute'
]

# Select categorical features
categorical_columns = ['carmodel', 'carname', 'platform']

# Prepare the data
X_numerical = df[feature_columns].copy()
X_categorical = df[categorical_columns].copy()
y = df['is_done']

# Encode categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X_categorical[col] = le.fit_transform(X_categorical[col].astype(str))
    label_encoders[col] = le

# Combine numerical and categorical features
X = pd.concat([X_numerical, X_categorical], axis=1)

print("\nFeature matrix shape:", X.shape)
print("Target vector shape:", y.shape)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

numerical_features = feature_columns
X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

# Build the neural network model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Display model architecture
model.summary()

# Add early stopping with more patience
from tensorflow.keras import callbacks

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,  # больше терпения
    restore_best_weights=True,
    verbose=1
)

# Train the model
history = model.fit(X_train_scaled, y_train,
                    epochs=100,  # увеличим лимит
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Make predictions
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int)

# Print evaluation metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"\nROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Function to find optimal price for a new order
def find_optimal_price(model, scaler, base_features, price_candidates, label_encoders):
    """
    Find the optimal price that maximizes expected revenue.
    
    Parameters:
    model: trained neural network model
    scaler: fitted StandardScaler
    base_features: dict with base feature values
    price_candidates: list of price candidates to evaluate
    label_encoders: dict of fitted label encoders for categorical variables
    
    Returns:
    optimal_price: price that maximizes expected revenue
    max_revenue: maximum expected revenue
    """
    max_revenue = 0
    optimal_price = price_candidates[0]
    
    # Create a base dataframe with one row
    base_df = pd.DataFrame([base_features])
    
    for price in price_candidates:
        # Update price-related features
        temp_df = base_df.copy()
        temp_df['price_bid_local'] = price
        temp_df['price_increase_percent'] = (price - temp_df['price_start_local']) / temp_df['price_start_local']
        temp_df['price_abs_diff'] = price - temp_df['price_start_local']
        temp_df['price_per_km'] = price / (temp_df['distance_km'] + 1e-5)
        temp_df['price_per_minute'] = price / (temp_df['duration_minutes'] + 1e-5)
        
        # Prepare features in the same way as training data
        X_numerical = temp_df[feature_columns].copy()
        X_categorical = temp_df[categorical_columns].copy()
        
        # Encode categorical variables
        for col in categorical_columns:
            if col in label_encoders:
                # Handle unseen labels
                X_categorical[col] = X_categorical[col].apply(
                    lambda x: x if x in label_encoders[col].classes_ else 'unknown'
                )
                # Update classes if needed
                if 'unknown' not in label_encoders[col].classes_:
                    label_encoders[col].classes_ = np.append(label_encoders[col].classes_, 'unknown')
                X_categorical[col] = label_encoders[col].transform(X_categorical[col])
        
        # Combine features
        X_temp = pd.concat([X_numerical, X_categorical], axis=1)
        
        # Scale numerical features
        X_temp_scaled = X_temp.copy()
        X_temp_scaled[numerical_features] = scaler.transform(X_temp[numerical_features])
        
        # Predict probability
        prob = model.predict(X_temp_scaled, verbose=0)[0][0]
        
        # Calculate expected revenue
        expected_revenue = price * prob
        
        print(f"Price: {price}, Probability: {prob:.4f}, Expected Revenue: {expected_revenue:.2f}")
        
        if expected_revenue > max_revenue:
            max_revenue = expected_revenue
            optimal_price = price
    
    return optimal_price, max_revenue

# Example usage of the price optimization function
print("\n" + "="*50)
print("PRICE OPTIMIZATION EXAMPLE")
print("="*50)

# Create example base features (you would replace this with actual order data)
example_base_features = {
    'distance_km': 3.0,  # converted to km
    'duration_minutes': 10.0,  # converted to minutes
    'driver_rating': 4.8,
    'pickup_distance_km': 0.5,  # converted to km
    'pickup_duration_minutes': 2.0,  # converted to minutes
    'price_start_local': 200,
    'carmodel': 'Logan',
    'carname': 'Renault',
    'platform': 'android'
}

# Define price candidates to evaluate
price_candidates = [150, 180, 200, 220, 250, 280, 300, 320, 350]

print("Base order features:")
for key, value in example_base_features.items():
    print(f"  {key}: {value}")

print(f"\nEvaluating price candidates: {price_candidates}")

optimal_price, max_revenue = find_optimal_price(
    model, scaler, example_base_features, price_candidates, label_encoders
)

print(f"\nOptimal Price: {optimal_price}")
print(f"Maximum Expected Revenue: {max_revenue:.2f}")

# Save the model
model.save('pricing_model.h5')
print("\nModel saved as 'pricing_model.h5'")

print("\nAnalysis complete!")