import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model

# === CONFIGURAÇÃO ===

MODEL_BASE_LOCAL = "smartparknet_dashboard/backend/modelos"
MODEL_BASE_FED = "smartparknet_dashboard/backend/modelos_federated"
TEST_CSV = "datasets/teste_comum.csv"
SEQ_LEN = 5

def preparar_dados_teste(df):
    data = df['status'].values
    X, y = [], []
    for i in range(len(data) - SEQ_LEN):
        X.append(data[i:i+SEQ_LEN])
        y.append(data[i+SEQ_LEN])
    X = np.array(X)
    y = np.array(y)
    return X.reshape((X.shape[0], SEQ_LEN, 1)), y

def avaliar_modelo(model_path, X, y):
    model = load_model(model_path)
    y_pred = model.predict(X).flatten()
    return {
        "MAE": mean_absolute_error(y, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
        "R2": r2_score(y, y_pred)
    }

# === INÍCIO ===

df_test = pd.read_csv(TEST_CSV)
X_test, y_test = preparar_dados_teste(df_test)

print("\n📊 Comparação de Modelos Locais vs Federated\n")

for camara in os.listdir(MODEL_BASE_LOCAL):
    camara_local = os.path.join(MODEL_BASE_LOCAL, camara, "b")
    camara_fed = os.path.join(MODEL_BASE_FED, camara)

    if not os.path.isdir(camara_local) or not os.path.isdir(camara_fed):
        continue

    for nome in os.listdir(camara_local):
        if nome.endswith(".h5"):
            semana = nome.split("_")[2].replace(".h5", "")
            local_path = os.path.join(camara_local, nome)
            fed_name = f"{camara}_b_{semana}_federated_model.h5"
            fed_path = os.path.join(camara_fed, fed_name)

            if not os.path.exists(fed_path):
                print(f"⚠️ Faltou modelo federado para {fed_name}")
                continue

            r_local = avaliar_modelo(local_path, X_test, y_test)
            r_fed = avaliar_modelo(fed_path, X_test, y_test)

            print(f"\n📁 {camara} — Semana {semana}")
            print(f"   Local     → MAE: {r_local['MAE']:.4f}  RMSE: {r_local['RMSE']:.4f}  R²: {r_local['R2']:.4f}")
            print(f"   Federated → MAE: {r_fed['MAE']:.4f}  RMSE: {r_fed['RMSE']:.4f}  R²: {r_fed['R2']:.4f}")
