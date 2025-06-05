import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import flwr as fl

# ⚙️ Configurações
CLIENT_ID = os.environ.get("CLIENT_ID", "033")
CSV_PATH = f"../smartparknet_dashboard/backend/dados_C\u00e2mara_{CLIENT_ID}.csv"
SEQUENCE_LENGTH = 5

# 📊 Função para carregar dados locais
def load_data(sequence_length=5):
    df = pd.read_csv(CSV_PATH)
    df['recvTime'] = pd.to_datetime(df['recvTime'])
    df = df.sort_values('recvTime')
    df['status'] = df['status'].map({'free': 0, 'occupied': 1})

    data = df['status'].values
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = X.reshape((X.shape[0], sequence_length, 1))

    return X, y

# 🧠 Modelo base
def build_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(SEQUENCE_LENGTH, 1)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# 🌐 Classe Flower Client
class SmartParkingClient(fl.client.NumPyClient):
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.X, self.y, epochs=int(config.get("epochs", 1)), batch_size=int(config.get("batch_size", 32)), verbose=0)
        return self.model.get_weights(), len(self.X), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X, self.y, verbose=0)
        return loss, len(self.X), {"accuracy": accuracy}

# 🚀 Iniciar cliente
if __name__ == "__main__":
    X, y = load_data(SEQUENCE_LENGTH)
    model = build_model()
    client = SmartParkingClient(model, X, y)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
