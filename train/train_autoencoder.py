import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import joblib

df = pd.read_csv("./csv_files/normal.csv")
df_numeric = df.select_dtypes(include=['float64','int64']).drop(columns=['time'], errors='ignore')

# 스케일링
scaler = StandardScaler()
normal_scaled = scaler.fit_transform(df_numeric)

# Autoencoder 정의
input_dim = normal_scaled.shape[1]
input_layer = Input(shape=(input_dim,))

# 인코더
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)
encoded = Dense(8, activation='relu')(encoded)  # encoding_dim = 8

# 디코더
decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(decoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# 학습
autoencoder.fit(
    normal_scaled, normal_scaled,
    epochs=30,         # 최적 epochs
    batch_size=16,     # 최적 batch_size
    verbose=0
)

# Autoencoder + Scaler + 컬럼 정보 저장
joblib.dump({
    "scaler": scaler,
    "feature_columns": df_numeric.columns.tolist()
}, "../models/autoencoder_scaler_columns.pkl")


# 저장 (모델은 h5, 스케일러는 따로 pkl)
autoencoder.save("../models/autoencoder.h5")
print("✅ Autoencoder + Scaler 저장 완료")

