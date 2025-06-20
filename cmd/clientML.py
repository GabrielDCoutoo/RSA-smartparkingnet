import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import flwr as fl
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json

# ============================== CONFIG ==============================
DATASET_PATH = os.environ.get("DATASET")
if not DATASET_PATH:
    raise ValueError("Variável de ambiente DATASET não definida")

PLOT_PERFORMANCE = os.environ.get("PLOT_PERFORMANCE", "false").lower() == "true"
EPOCHS = int(os.environ.get("EPOCHS", 10))  # Increased default epochs
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))

print(f"📊 A processar dataset: {DATASET_PATH}")

# ============================== LOAD DATA ==============================
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset carregado: {len(df)} amostras")
    print("Colunas disponíveis:", df.columns)
except Exception as e:
    print(f"Erro ao carregar dataset: {e}")
    exit(1)

if 'status' not in df.columns or 'timestamp_seconds' not in df.columns:
    raise ValueError("Colunas 'status' e/ou 'timestamp_seconds' não encontradas.")

# ============================== IMPROVED FEATURE ENGINEERING ==============================
def create_enhanced_features(df):
    """Create comprehensive features for parking prediction"""
    df = df.copy()
    
    # Convert timestamp to datetime (fix the double conversion issue)
    start_time = pd.to_datetime('2025-02-26')  # Your data start date
    df['datetime'] = start_time + pd.to_timedelta(df['timestamp_seconds'], unit='s')
    
    # Time-based features
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['weekday'] = df['datetime'].dt.weekday  # 0=Monday, 6=Sunday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['day_of_month'] = df['datetime'].dt.day
    
    # Parking-specific time features
    df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(int)
    df['is_business_hours'] = (df['hour'].between(8, 18) & (df['weekday'] < 5)).astype(int)
    df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
    df['is_lunch_time'] = (df['hour'].between(11, 14)).astype(int)
    
    # Cyclical encoding for time features (important for neural networks)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    
    # Temporal features (trend)
    df['time_normalized'] = (df['timestamp_seconds'] - df['timestamp_seconds'].min()) / (df['timestamp_seconds'].max() - df['timestamp_seconds'].min())
    
    # Lag features (if data is sequential)
    df = df.sort_values('timestamp_seconds')
    df['prev_status'] = df['status'].shift(1).fillna(df['status'].iloc[0])
    df['status_change'] = (df['status'] != df['prev_status']).astype(int)
    
    # Moving averages (occupancy trends)
    window_sizes = [5, 10, 20]
    for window in window_sizes:
        if len(df) > window:
            df[f'occupancy_ma_{window}'] = df['status'].rolling(window=window, min_periods=1).mean()
    
    return df

# Apply enhanced feature engineering
df = create_enhanced_features(df)

# ============================== DATA PREPARATION ==============================
# Target variable
y = df['status'].astype(int).values

# Feature selection (remove non-predictive columns)
feature_columns = [
    'timestamp_seconds', 'hour', 'minute', 'weekday', 'day_of_month',
    'is_weekend', 'is_rush_hour', 'is_business_hours', 'is_night', 'is_lunch_time',
    'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'time_normalized',
    'prev_status', 'status_change'
]

# Add moving average features if they exist
for col in df.columns:
    if col.startswith('occupancy_ma_'):
        feature_columns.append(col)

X = df[feature_columns].values.astype(np.float32)

# Improved normalization using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"Shape dos dados: X={X.shape}, y={y.shape}")
print(f"Classes únicas: {np.unique(y)}")
print(f"Distribuição de classes: {np.bincount(y)}")

# Check class balance
class_ratio = np.bincount(y)
if len(class_ratio) == 2:
    minority_ratio = min(class_ratio) / sum(class_ratio)
    print(f"Ratio da classe minoritária: {minority_ratio:.3f}")
    if minority_ratio < 0.1:
        print("⚠️ Dataset muito desbalanceado. Considere SMOTE ou outras técnicas de balanceamento.")

if len(X) < 10:
    print(f"⚠️ Dataset muito pequeno ({len(X)} amostras). Mínimo recomendado: 10")
    exit(1)

# Stratified split with better test size calculation
test_size = max(0.2, min(0.3, 30 / len(X)))
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

print(f"Dados de treino: {len(x_train)}, Dados de teste: {len(x_test)}")

# ============================== IMPROVED MODEL ==============================
def build_model():
    """Build an improved model architecture"""
    model = keras.Sequential([
        keras.layers.Input(shape=(x_train.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    
    # Use a lower learning rate for more stable training
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer, 
        loss="binary_crossentropy", 
        metrics=["accuracy", "precision", "recall"]
    )
    return model

# Load or create model
MODEL_GLOBAL_PATH = "RSA/model_global/model_global.h5"

if os.path.exists(MODEL_GLOBAL_PATH):
    print(f"📥 A carregar modelo global de {MODEL_GLOBAL_PATH}")
    model = tf.keras.models.load_model(MODEL_GLOBAL_PATH)
else:
    print("⚠️ Modelo global não encontrado. A criar novo modelo de raiz.")
    model = build_model()

print("✅ Modelo criado")

# ============================== FEDERATED CLIENT ==============================
class FederatedClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        print(f"🔄 A treinar com {len(x_train)} amostras...")
        model.set_weights(parameters)

        # Enhanced class weights calculation
        if len(np.unique(y_train)) > 1:
            class_weights = class_weight.compute_class_weight(
                class_weight="balanced",
                classes=np.unique(y_train),
                y=y_train
            )
            cw_dict = dict(enumerate(class_weights))
            print(f"Class weights: {cw_dict}")
        else:
            cw_dict = None

        # Add callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=3, 
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=2, 
                min_lr=1e-6
            )
        ]

        history = model.fit(
            x_train, y_train,
            epochs=EPOCHS,
            batch_size=min(BATCH_SIZE, len(x_train)),
            validation_data=(x_test, y_test),
            class_weight=cw_dict,
            callbacks=callbacks,
            verbose=1
        )

        # Detailed evaluation
        y_pred = (model.predict(x_test) > 0.5).astype(int)
        print("\n📊 Relatório de Classificação:")
        print(classification_report(y_test, y_pred, target_names=['Free', 'Occupied']))
        
        # Save model
        client_id = os.path.basename(DATASET_PATH).split("_")[2]
        model_dir = "smartparknet_dashboard/backend/modelos_federated/"
        os.makedirs(model_dir, exist_ok=True)
        model.save(f"{model_dir}/client_{client_id}.h5")
        print(f"💾 Modelo local guardado: {model_dir}/client_{client_id}.h5")

        if PLOT_PERFORMANCE:
            plot_performance(history)

        # Save training history
        history_dict = history.history
        client_id = os.environ.get("DATASET", "unknown").split("/")[-1].replace(".csv", "")
        output_dir = "histories"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/history_{client_id}.json", "w") as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history_dict.items()}, f)

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

        loss, accuracy, precision, recall = model.evaluate(x_test, y_test, verbose=0)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        print(f"📊 Avaliação - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f"📊 Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")
        
        return loss, len(x_test), {
            "accuracy": accuracy, 
            "precision": precision, 
            "recall": recall,
            "f1_score": f1_score
        }

# ============================== ENHANCED PLOTTING ==============================
def plot_performance(history):
    dataset_name = os.path.basename(DATASET_PATH).replace('.csv', '')

    plt.figure(figsize=(15, 10))

    # Accuracy plot
    plt.subplot(2, 3, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy", marker='o')
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="Val Accuracy", marker='s')
    plt.title(f"Model Accuracy - {dataset_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss plot
    plt.subplot(2, 3, 2)
    plt.plot(history.history["loss"], label="Train Loss", marker='o')
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val Loss", marker='s')
    plt.title(f"Model Loss - {dataset_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Precision plot
    plt.subplot(2, 3, 3)
    if "precision" in history.history:
        plt.plot(history.history["precision"], label="Train Precision", marker='o')
    if "val_precision" in history.history:
        plt.plot(history.history["val_precision"], label="Val Precision", marker='s')
    plt.title(f"Model Precision - {dataset_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Recall plot
    plt.subplot(2, 3, 4)
    if "recall" in history.history:
        plt.plot(history.history["recall"], label="Train Recall", marker='o')
    if "val_recall" in history.history:
        plt.plot(history.history["val_recall"], label="Val Recall", marker='s')
    plt.title(f"Model Recall - {dataset_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Learning rate (if available)
    plt.subplot(2, 3, 5)
    if "lr" in history.history:
        plt.plot(history.history["lr"], label="Learning Rate", marker='o')
        plt.title(f"Learning Rate - {dataset_name}")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
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