import pandas as pd
import numpy as np

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

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Basic statistics
print("\nBasic statistics for numerical columns:")
print(df.describe())

# Convert target variable to binary (0 for cancel, 1 for done)
df['is_done_binary'] = df['is_done'].map({'cancel': 0, 'done': 1})

# Check the conversion
print("\nBinary target variable distribution:")
print(df['is_done_binary'].value_counts())

# Price analysis
print("\nPrice analysis:")
print("Price start local statistics:")
print(df['price_start_local'].describe())
print("\nPrice bid local statistics:")
print(df['price_bid_local'].describe())

# Calculate price differences
df['price_diff'] = df['price_bid_local'] - df['price_start_local']
df['price_increase_percent'] = (df['price_bid_local'] - df['price_start_local']) / df['price_start_local'] * 100

print("\nPrice difference statistics:")
print(df['price_diff'].describe())
print("\nPrice increase percentage statistics:")
print(df['price_increase_percent'].describe())

# Correlation between price difference and acceptance
print("\nCorrelation between price difference and acceptance:")
correlation = df['price_diff'].corr(df['is_done_binary'])
print(f"Correlation: {correlation:.4f}")

print("\nCorrelation between price increase percentage and acceptance:")
correlation = df['price_increase_percent'].corr(df['is_done_binary'])
print(f"Correlation: {correlation:.4f}")

# Acceptance rate by price increase percentage ranges
df['price_increase_range'] = pd.cut(df['price_increase_percent'], 
                                   bins=[-np.inf, -10, 0, 10, 20, 30, np.inf], 
                                   labels=['<-10%', '-10% to 0%', '0% to 10%', '10% to 20%', '20% to 30%', '>30%'])

print("\nAcceptance rate by price increase percentage ranges:")
acceptance_by_range = df.groupby('price_increase_range')['is_done_binary'].agg(['count', 'sum', 'mean'])
acceptance_by_range.columns = ['Total', 'Accepted', 'Acceptance Rate']
print(acceptance_by_range)

# Distance and duration analysis
print("\nDistance and duration analysis:")
print("Distance in meters statistics:")
print(df['distance_in_meters'].describe())
print("\nDuration in seconds statistics:")
print(df['duration_in_seconds'].describe())

# Driver rating analysis
print("\nDriver rating analysis:")
print("Driver rating statistics:")
print(df['driver_rating'].describe())

# Platform distribution
print("\nPlatform distribution:")
print(df['platform'].value_counts())

# Car model distribution (top 10)
print("\nTop 10 car models:")
print(df['carmodel'].value_counts().head(10))

# Car name distribution (top 10)
print("\nTop 10 car names:")
print(df['carname'].value_counts().head(10))

print("\nData exploration complete!")