import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the data
df = pd.read_csv('train.csv')

# Convert target variable to binary (0 for cancel, 1 for done)
df['is_done_binary'] = df['is_done'].map({'cancel': 0, 'done': 1})

# Calculate price differences
df['price_diff'] = df['price_bid_local'] - df['price_start_local']
df['price_increase_percent'] = (df['price_bid_local'] - df['price_start_local']) / df['price_start_local'] * 100

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Ride Order Data Analysis', fontsize=16)

# 1. Distribution of price increase percentage
axes[0, 0].hist(df['price_increase_percent'], bins=50, alpha=0.7, color='skyblue')
axes[0, 0].set_xlabel('Price Increase Percentage (%)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Price Increase Percentage')
axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7, label='No Price Change')
axes[0, 0].legend()

# 2. Acceptance rate by price increase percentage ranges
df['price_increase_range'] = pd.cut(df['price_increase_percent'], 
                                   bins=[-np.inf, -10, 0, 10, 20, 30, np.inf], 
                                   labels=['<-10%', '-10% to 0%', '0% to 10%', '10% to 20%', '20% to 30%', '>30%'])

acceptance_by_range = df.groupby('price_increase_range')['is_done_binary'].mean()
acceptance_by_range.plot(kind='bar', ax=axes[0, 1], color='lightgreen')
axes[0, 1].set_xlabel('Price Increase Percentage Range')
axes[0, 1].set_ylabel('Acceptance Rate')
axes[0, 1].set_title('Acceptance Rate by Price Increase Percentage')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Distribution of ride distances
axes[1, 0].hist(df['distance_in_meters'] / 1000, bins=50, alpha=0.7, color='orange')
axes[1, 0].set_xlabel('Distance (km)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Ride Distances')

# 4. Distribution of driver ratings
axes[1, 1].hist(df['driver_rating'], bins=30, alpha=0.7, color='purple')
axes[1, 1].set_xlabel('Driver Rating')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of Driver Ratings')

plt.tight_layout()
plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Show correlation between key numerical features and acceptance
numerical_features = ['distance_in_meters', 'duration_in_seconds', 'driver_rating', 
                     'pickup_in_meters', 'pickup_in_seconds', 'price_start_local', 
                     'price_bid_local', 'price_diff', 'price_increase_percent']

correlations = df[numerical_features + ['is_done_binary']].corr()['is_done_binary'].drop('is_done_binary').sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=correlations.values, y=correlations.index, palette='viridis')
plt.title('Correlation of Features with Order Acceptance')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.savefig('feature_correlations.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualizations saved as 'data_analysis.png' and 'feature_correlations.png'")
print("\nTop correlations with order acceptance:")
print(correlations.head(10))