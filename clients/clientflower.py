import os
import numpy as np
import pandas as pd
import flwr as fl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# ⚙️ CONFIGURAÇÃO
SEQUENCE_LENGTH = 5
DATASET = os.environ.get("DATASET")

if not DATASET or not os.path.exists(DATASET):
    raise ValueError("❌ Caminho inválido ou variável de ambiente DATASET não definida.")

# 📊 CARREGAR E PREPARAR DADOS
def load_data(sequence_length):
    df = pd.read_csv(DATASET)
    df['recvTime'] = pd.to_datetime(df['recvTime'])
    df = df.sort_values('recvTime')
    df['status'] = df['status'].map({'free': 0, 'occupied': 1})

    data = df['status'].dropna().values
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = X.reshape((X.shape[0], sequence_length, 1))

    return train_test_split(X, y, test_size=0.3, random_state=42)

# 🧠 DEFINIR MODELO
def build_model(input_shape):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_shape,)),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# 🌐 CLIENTE FLOWER
class SmartParkingClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=int(config.get("epochs", 1)),
                       batch_size=int(config.get("batch_size", 32)), verbose=0)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}

# 🚀 INICIAR CLIENTE
if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data(SEQUENCE_LENGTH)
    model = build_model((SEQUENCE_LENGTH, 1))
    client = SmartParkingClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
