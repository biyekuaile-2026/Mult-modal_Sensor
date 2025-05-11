import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import joblib

# Step 1: Load data
file_path = r'Pressure data_data.xlsx'  # Modify training data path
data = pd.read_excel(file_path)

# Assume the column names are 'T(℃)', 'Cp(F)', 'Pressure(kPa)'
X = data[['T(℃)', 'Cp(F)']]
y = data['Pressure(kPa)']

# Step 2: Data preprocessing
# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Extract time-series features (e.g., lag values and moving averages)
X = pd.DataFrame(X_imputed, columns=['T(℃)', 'Cp(F)'])
X['T_lag1'] = X['T(℃)'].shift(1)
X['Cp_lag1'] = X['Cp(F)'].shift(1)
X['T_ma3'] = X['T(℃)'].rolling(window=3).mean()
X['Cp_ma3'] = X['Cp(F)'].rolling(window=3).mean()
X.fillna(method='bfill', inplace=True)  # Backfill missing values

# Generate polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Standardize features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_poly)

# Standardize target variable
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(np.array(y).reshape(-1, 1))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# LSTM model expects input shape: (samples, timesteps, features)
X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Step 3: Define and train LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(128, activation='relu', return_sequences=True), input_shape=(1, X_train.shape[1])))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64, activation='relu', return_sequences=True)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(LSTM(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(1, kernel_regularizer='l2'))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=75, batch_size=64, validation_split=0.2, verbose=1)

# Step 4: Evaluate and predict
y_pred_scaled = model.predict(X_test_reshaped)
y_pred = scaler_y.inverse_transform(y_pred_scaled)  # Inverse transform predictions

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred_scaled)
mae = mean_absolute_error(y_test, y_pred_scaled)
r2 = r2_score(y_test, y_pred_scaled)

# Calculate Adjusted R²
n = len(y_test)  # number of samples
p = X_test.shape[1]  # number of features
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print('\nMSE: {:.4f}'.format(mse))
print('MAE: {:.4f}'.format(mae))
print('R²: {:.4f}'.format(r2))
print('Adjusted R²: {:.4f}'.format(r2_adj))

# Step 5: RMSLE
def rmsle(y_true, y_pred):
    # Ensure y_true and y_pred are non-negative
    y_true = np.maximum(y_true, 0)  # Replace negative values with 0
    y_pred = np.maximum(y_pred, 0)  # Replace negative values with 0
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

rmsle_value = rmsle(y_test, y_pred)
print(f'RMSLE: {rmsle_value:.4f}')

# Step 6: 2D plots with colorbar for Cp values
X_test_original = scaler_X.inverse_transform(X_test)  # Inverse transform features
y_test_original = scaler_y.inverse_transform(y_test)  # Inverse transform actual values
y_pred_original = scaler_y.inverse_transform(y_pred_scaled)  # Inverse transform predictions
