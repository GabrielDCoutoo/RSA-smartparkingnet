import pandas as pd
import numpy as np
import os
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import resample

SEQUENCE_LENGTH = 5
OUTPUT_BASE = "smartparknet_dashboard/backend/modelos"

# Limite mínimo de dados por entityId
MIN_DATA_POINTS = 1000

def preparar_dados_entity(grupo, sequence_length=SEQUENCE_LENGTH, balancear=False):
    grupo = grupo.sort_values('recvTime')
    grupo['status'] = grupo['status'].map({'free': 0, 'occupied': 1})

    if balancear:
        df_majority = grupo[grupo.status == 1]
        df_minority = grupo[grupo.status == 0]
        if len(df_minority) == 0 or len(df_majority) == 0:
            return None, None, None, None
        df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
        grupo = pd.concat([df_majority_downsampled, df_minority])
        grupo = grupo.sort_values('recvTime')

    data = grupo['status'].values
    if len(data) <= sequence_length:
        return None, None, None, None

    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = X.reshape((X.shape[0], sequence_length, 1))

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_lstm_model(X_train, y_train, sequence_length=SEQUENCE_LENGTH):
    model = Sequential()
    model.add(LSTM(64, input_shape=(sequence_length, 1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    return model

def extrair_info_nome_ficheiro(nome):
    partes = nome.replace(".csv", "").split("_")
    camara = "_".join(partes[1:3])
    tipo = partes[3]
    semana = partes[4]
    return camara, tipo, semana

def main():
    if len(sys.argv) < 2:
        print("⚠️  Usa: python train_LSTM.py <ficheiro_csv>")
        sys.exit(1)

    caminho_csv = sys.argv[1]
    if not os.path.isfile(caminho_csv):
        print(f"❌ Ficheiro não encontrado: {caminho_csv}")
        sys.exit(1)

    print(f"📁 A carregar dados de {caminho_csv}...")
    df = pd.read_csv(caminho_csv)
    df['recvTime'] = pd.to_datetime(df['recvTime'])

    if 'entityId' not in df.columns:
        print("❌ Coluna 'entityId' em falta no CSV.")
        sys.exit(1)

    nome_ficheiro = os.path.basename(caminho_csv)
    camara, tipo, semana = extrair_info_nome_ficheiro(nome_ficheiro)

    for entity_id, grupo in df.groupby('entityId'):
        print(f"🔍 A preparar treino para entityId {entity_id}...")

        if len(grupo) < MIN_DATA_POINTS:
            print(f"⏩ Ignorado (apenas {len(grupo)} pontos)")
            continue

        X_train, X_test, y_train, y_test = preparar_dados_entity(grupo)
        if X_train is None:
            print("⚠️  Dados insuficientes após preparação.")
            continue

        print("🚀 A treinar modelo LSTM...")
        model = train_lstm_model(X_train, y_train)

        pasta_saida = os.path.join(OUTPUT_BASE, camara, tipo, entity_id)
        os.makedirs(pasta_saida, exist_ok=True)

        modelo_path = os.path.join(pasta_saida, f"modelo_{tipo}_{semana}.h5")
        relatorio_path = os.path.join(pasta_saida, f"relatorio_{tipo}_{semana}.txt")

        model.save(modelo_path)
        print(f"💾 Modelo guardado em {modelo_path}")

        print("🧪 A avaliar modelo...")
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        report = classification_report(y_test, y_pred, zero_division=0)

        with open(relatorio_path, "w") as f:
            f.write(report)
        print(f"📝 Relatório guardado em {relatorio_path}")

if __name__ == "__main__":
    main()
