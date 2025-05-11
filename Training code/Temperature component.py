import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Read the Excel file
df = pd.read_excel(r'Temperature data.xlsx')

# Extract Z and T columns
Z = df['Z (Ω)'].values
T = df['T(℃)'].values

# Standardize input data
scaler_Z = StandardScaler()
Z_scaled = scaler_Z.fit_transform(Z.reshape(-1, 1))

scaler_T = StandardScaler()
T_scaled = scaler_T.fit_transform(T.reshape(-1, 1))

# Split into training and validation sets
Z_train, Z_val, T_train, T_val = train_test_split(Z_scaled, T_scaled, test_size=0.2, random_state=42)

# Reshape into LSTM input format: (samples, timesteps, features)
Z_train = Z_train.reshape((Z_train.shape[0], 1, Z_train.shape[1]))
Z_val = Z_val.reshape((Z_val.shape[0], 1, Z_val.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(Z_train.shape[1], Z_train.shape[2]), return_sequences=True))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(Z_train, T_train, epochs=50, validation_data=(Z_val, T_val))

# Predict using the model
T_pred = model.predict(Z_val)

# Inverse transform the standardized values
Z_val_original = scaler_Z.inverse_transform(Z_val.reshape(-1, 1))
T_val_original = scaler_T.inverse_transform(T_val.reshape(-1, 1))
T_pred_original = scaler_T.inverse_transform(T_pred.reshape(-1, 1))

# Compute evaluation metrics (based on standardized values)
mae = mean_absolute_error(T_val, T_pred)
mse = mean_squared_error(T_val, T_pred)
r2 = r2_score(T_val, T_pred)

# Calculate Adjusted R²
n = len(T_val)  # number of samples
p = 1  # number of features (only Z is used here)
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)  # formula for Adjusted R²

# Compute RMSLE (on original values, must be non-negative)
T_val_clipped = np.clip(T_val_original, a_min=0, a_max=None)
T_pred_clipped = np.clip(T_pred_original, a_min=0, a_max=None)
rmsle = np.sqrt(mean_squared_log_error(T_val_clipped, T_pred_clipped))

# Print evaluation metrics
print('Mean Absolute Error (MAE): {:.4f}'.format(mae))
print('Mean Squared Error (MSE): {:.4f}'.format(mse))
print('R^2 Score: {:.4f}'.format(r2))
print('Adjusted R^2: {:.4f}'.format(r2_adj))
print('Root Mean Squared Log Error (RMSLE): {:.4f}'.format(rmsle))