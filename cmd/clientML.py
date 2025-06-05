import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import flwr as fl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

# ============================== CONFIG ==============================
DATASET_PATH = os.environ.get("DATASET")
if not DATASET_PATH:
    raise ValueError("Variável de ambiente DATASET não definida")

PLOT_PERFORMANCE = os.environ.get("PLOT_PERFORMANCE", "false").lower() == "true"
EPOCHS = int(os.environ.get("EPOCHS", 5))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))

print(f"📊 A processar dataset: {DATASET_PATH}")

# ============================== LOAD DATA ==============================
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset carregado: {len(df)} amostras")
    print(f"Colunas disponíveis: {list(df.columns)}")
except Exception as e:
    print(f"Erro ao carregar dataset: {e}")
    exit(1)

# Verificar se temos as colunas necessárias
if 'status' not in df.columns:
    raise ValueError("A coluna 'status' não foi encontrada no dataset.")

if 'timestamp_seconds' not in df.columns:
    raise ValueError("A coluna 'timestamp_seconds' não foi encontrada no dataset.")

# Preparar dados
y = df['status'].astype(int).values
X = df[['timestamp_seconds']].values.astype(np.float32)

# Normalizar timestamp_seconds
X = (X - X.min()) / (X.max() - X.min() + 1e-8)

print(f"Shape dos dados: X={X.shape}, y={y.shape}")
print(f"Classes únicas: {np.unique(y)}")

# Verificar se temos dados suficientes
if len(X) < 10:
    print(f"⚠️ Dataset muito pequeno ({len(X)} amostras). Mínimo recomendado: 10")
    exit(1)

# Split dos dados
test_size = max(0.2, min(0.4, 20 / len(X)))  # Adaptar test_size ao tamanho do dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

print(f"Dados de treino: {len(x_train)}, Dados de teste: {len(x_test)}")

# ============================== BUILD MODEL ==============================
def build_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model = build_model()
print("✅ Modelo criado")

# ============================== FEDERATED CLIENT ==============================
class FederatedClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        print(f"🔄 A treinar com {len(x_train)} amostras...")
        model.set_weights(parameters)
        
        # Converter labels para categorical
        y_train_cat = to_categorical(y_train, 2)
        y_test_cat = to_categorical(y_test, 2)
        
        history = model.fit(
            x_train, y_train_cat,
            epochs=EPOCHS,
            batch_size=min(BATCH_SIZE, len(x_train)),
            validation_data=(x_test, y_test_cat),
            verbose=1
        )

        if PLOT_PERFORMANCE:
            plot_performance(history)

        final_loss = history.history["loss"][-1]
        final_accuracy = history.history["accuracy"][-1]
        
        print(f"✅ Treino concluído - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")

        return model.get_weights(), len(x_train), {
            "loss": final_loss,
            "accuracy": final_accuracy
        }

    def evaluate(self, parameters, config):
        print("🧪 A avaliar modelo...")
        model.set_weights(parameters)
        
        y_test_cat = to_categorical(y_test, 2)
        loss, accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
        
        # Fazer predições para métricas adicionais
        y_pred = model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print(f"📊 Avaliação - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return loss, len(x_test), {"accuracy": accuracy}

# ============================== PLOT ==============================
def plot_performance(history):
    dataset_name = os.path.basename(DATASET_PATH).replace('.csv', '')
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy", marker='o')
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="Val Accuracy", marker='s')
    plt.title(f"Model Accuracy - {dataset_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss", marker='o')
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val Loss", marker='s')
    plt.title(f"Model Loss - {dataset_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Salvar com nome único
    plot_filename = f"performance_{dataset_name}_{int(time.time())}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"📈 Gráfico salvo: {plot_filename}")
    plt.close()

# ============================== MAIN ==============================
if __name__ == "__main__":
    print(f"🌐 A conectar ao servidor em localhost:8080...")
    print(f"📂 Dataset: {DATASET_PATH}")
    
    try:
        client = FederatedClient()
        fl.client.start_numpy_client(
            server_address="localhost:8080", 
            client=client
        )
    except Exception as e:
        print(f"❌ Erro na conexão: {e}")
        print("Certifique-se de que o servidor está a correr!")