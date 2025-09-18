import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import joblib

df = pd.read_csv("./csv_files/train_anomaly.csv")
df_numeric = df.select_dtypes(include=['float64','int64']).drop(columns=['time'], errors='ignore')

normal_train, _ = train_test_split(df_numeric, test_size=0.1, random_state=42)

# 스케일링
scaler = StandardScaler()
normal_scaled = scaler.fit_transform(normal_train)

# Autoencoder 정의
input_dim = normal_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation='relu')(input_layer)
encoded = Dense(8, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(normal_scaled, normal_scaled, epochs=50, batch_size=32, verbose=0)

# 저장 (모델은 h5, 스케일러는 따로 pkl)

autoencoder.save("../models/autoencoder.h5")
joblib.dump(scaler, "../models/autoencoder_scaler.pkl")
print("✅ Autoencoder + Scaler 저장 완료")

