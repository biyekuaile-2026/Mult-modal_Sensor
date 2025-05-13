import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import joblib

# STEP1
df1 = pd.read_excel(r'Strain data_without_strain.xlsx')
df2 = pd.read_excel(r'Strain data_with_strain')

T1 = df1['T(℃)'].values.reshape(-1, 1)
Z1 = df1['Z (Ω)'].values.reshape(-1, 1)

scaler_T = StandardScaler()
scaler_Z = StandardScaler()
T1 = scaler_T.fit_transform(T1)
Z1 = scaler_Z.fit_transform(Z1)

window_size = 3
T1_window = np.zeros((len(T1) - window_size + 1, window_size, 1))
Z1_window = np.zeros((len(Z1) - window_size + 1, window_size, 1))

for i in range(len(T1) - window_size + 1):
    T1_window[i] = T1[i:i + window_size]
    Z1_window[i] = Z1[i:i + window_size]

Z1_window = Z1_window[:, -1, :]

T1_train, T1_val, Z1_train, Z1_val = train_test_split(T1_window, Z1_window, test_size=0.2, random_state=42)

model1 = Sequential()
model1.add(LSTM(64, activation='relu', input_shape=(window_size, 1), return_sequences=True))
model1.add(LSTM(64, activation='relu'))
model1.add(Dense(1))

model1.compile(optimizer='adam', loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model1.fit(T1_train, Z1_train, epochs=100, validation_data=(T1_val, Z1_val))

Z1_pred_all = model1.predict(T1_window)

mae1 = mean_absolute_error(Z1_val, model1.predict(T1_val))
mse1 = mean_squared_error(Z1_val, model1.predict(T1_val))
r2_1 = r2_score(Z1_val, model1.predict(T1_val))

print('Model 1 - MAE: {:.4f}'.format(mae1))
print('Model 1 - MSE: {:.4f}'.format(mse1))
print('Model 1 - R^2: {:.4f}'.format(r2_1))

# STEP2
T2 = df2['T(℃)'].values
Z2 = df2['Z(Ω)'].values
Strain = df2['Strain'].values

T2_scaled = scaler_T.transform(T2.reshape(-1, 1))
T2_window = np.zeros((len(T2_scaled) - window_size + 1, window_size, 1))
for i in range(len(T2_scaled) - window_size + 1):
    T2_window[i] = T2_scaled[i:i + window_size]

Z1_pred_all_2 = model1.predict(T2_window).flatten()
Z_new = Z2[window_size - 1:] - Z1_pred_all_2

X = pd.DataFrame({'Temperature': T2[window_size - 1:], 'Impedance': Z_new})
y = Strain[window_size - 1:]

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_imputed)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

X_window = np.zeros((len(X_scaled) - window_size + 1, window_size, X_scaled.shape[1]))
for i in range(len(X_scaled) - window_size + 1):
    X_window[i] = X_scaled[i:i + window_size]

y_window = y[window_size - 1:]

X_train, X_test, y_train, y_test = train_test_split(X_window, y_window, test_size=0.2, random_state=42)

model = Sequential()
model.add(Bidirectional(LSTM(128, activation='relu', return_sequences=True), input_shape=(window_size, X_scaled.shape[1])))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64, activation='relu', return_sequences=True)))
model.add(Dropout(0.3))
model.add(LSTM(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, kernel_regularizer='l2'))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model.fit(X_train, y_train, epochs=200, batch_size=64, validation_split=0.2, verbose=1)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
n = len(y_test)
p = X_scaled.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
rmsle = np.sqrt(mean_squared_log_error(np.maximum(y_test, 0), np.maximum(y_pred.flatten(), 0)))

print('\nMSE: {:.4f}'.format(mse))
print('MAE: {:.4f}'.format(mae))
print('R²: {:.4f}'.format(r2))
print('Adjusted R²: {:.4f}'.format(adj_r2))
print('RMSLE: {:.4f}'.format(rmsle))

X_indices = np.arange(len(X_scaled) - window_size + 1)
_, X_test_indices, _, _ = train_test_split(X_indices, y_window, test_size=0.2, random_state=42)

T_test = T2[X_test_indices + window_size - 1]
Z_test = Z2[X_test_indices + window_size - 1]