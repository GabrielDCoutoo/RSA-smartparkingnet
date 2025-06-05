import flwr as fl
import pandas as pd
import numpy as np
import sys
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

class GhostClient(fl.client.NumPyClient):
    def __init__(self, csv_path, sequence_length=5):
        self.sequence_length = sequence_length
        self.csv_path = csv_path
        self.client_name = os.path.basename(csv_path).replace(".csv", "")
        self.X_train, self.y_train, self.X_test, self.y_test = self.load_data(csv_path)
        self.model = self.build_model()

    def load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df['recvTime'] = pd.to_datetime(df['recvTime'])
        df = df.sort_values('recvTime')
        df['status'] = df['status'].map({'free': 0, 'occupied': 1})
        data = df['status'].values

        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i+self.sequence_length])
            y.append(data[i+self.sequence_length])

        X = np.array(X)
        y = np.array(y)
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Dados insuficientes para treino.")
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        X = X.reshape((X.shape[0], self.sequence_length, 1))

        split = int(0.8 * len(X))
        return X[:split], y[:split], X[split:], y[split:]

    def build_model(self):
        model = Sequential()
        model.add(LSTM(64, input_shape=(self.sequence_length, 1)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(self.X_train, self.y_train, epochs=5, batch_size=32, verbose=0)
        loss = history.history["loss"][-1]

        output_dir = "smartparknet_dashboard/backend/modelos_federated"
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{self.client_name}_federated_model.h5")
        self.model.save(model_path)
        print(f"💾 Modelo federado guardado em {model_path}")

        return self.model.get_weights(), len(self.X_train), {"loss": loss}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        return loss, len(self.X_test), {"accuracy": accuracy}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Uso: python GhostClient_LSTM.py <caminho_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    nome = os.path.basename(csv_path).replace(".csv", "")
    print(f"👤 A iniciar GhostClient para ficheiro: {nome}")

    client = GhostClient(csv_path)
    fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())
