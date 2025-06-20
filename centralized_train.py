import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================== CONFIG ==============================
DATASET_PATH = "smartparknet_dashboard/backend/dados_centralizado.csv"
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 0.001

print("🚀 TREINAMENTO CENTRALIZADO MELHORADO")
print("=" * 50)

# ============================== LOAD DATA ==============================
df = pd.read_csv(DATASET_PATH)
print(f"Dataset carregado: {len(df)} amostras")
print(f"Colunas disponíveis: {list(df.columns)}")

# Check for missing essential columns
if 'status' not in df.columns:
    raise ValueError("Coluna 'status' não encontrada no dataset!")

# ============================== FEATURE ENGINEERING ==============================
def create_comprehensive_features(df):
    """Criar features abrangentes para predição de estacionamento"""
    df = df.copy()
    
    # 1. FEATURES TEMPORAIS
    if 'recvTime' in df.columns:
        df['datetime'] = pd.to_datetime(df['recvTime'], errors='coerce')
        print("✅ Usando coluna 'recvTime' para features temporais")
    elif 'timestamp_seconds' in df.columns:
        # Assumir que começa em uma data específica (ajustar conforme necessário)
        start_date = pd.Timestamp('2025-02-26')
        df['datetime'] = start_date + pd.to_timedelta(df['timestamp_seconds'], unit='s')
        print("✅ Usando coluna 'timestamp_seconds' para features temporais")
    else:
        print("⚠️ Nenhuma coluna de tempo encontrada, criando features temporais básicas")
        # Criar datetime usando o índice como aproximação
        df['datetime'] = pd.date_range(start='2025-01-01', periods=len(df), freq='1min')
    
    # Remove rows with invalid datetime
    df = df.dropna(subset=['datetime'])
    
    # Features de tempo
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['weekday'] = df['datetime'].dt.weekday  # 0=Monday, 6=Sunday
    df['day_of_month'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    
    # Features categóricas temporais
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18) & (df['weekday'] < 5)).astype(int)
    df['is_rush_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 9)) | 
                         ((df['hour'] >= 17) & (df['hour'] <= 19))).astype(int)
    df['is_lunch_time'] = ((df['hour'] >= 11) & (df['hour'] <= 14)).astype(int)
    df['is_night'] = ((df['hour'] <= 6) | (df['hour'] >= 22)).astype(int)
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] <= 12)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] <= 18)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] <= 22)).astype(int)
    
    # 2. ENCODING CÍCLICO (importante para redes neurais)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 3. FEATURES DE DISPOSITIVO/CÂMARA
    if 'refDevice' in df.columns:
        print("✅ Processando features de 'refDevice'")
        le_device = LabelEncoder()
        df['device_id'] = le_device.fit_transform(df['refDevice'].astype(str))
        
        # One-hot encoding para dispositivos (limitado para evitar dimensionalidade excessiva)
        unique_devices = df['refDevice'].nunique()
        print(f"Número de dispositivos únicos: {unique_devices}")
        
        if unique_devices <= 50:  # Só fazer one-hot se não for muitos dispositivos
            device_dummies = pd.get_dummies(df['refDevice'], prefix='device')
            df = pd.concat([df, device_dummies], axis=1)
            print(f"✅ One-hot encoding criado para {unique_devices} dispositivos")
        else:
            print(f"⚠️ Muitos dispositivos ({unique_devices}), usando apenas device_id numérico")
    
    # 4. FEATURES DE LOCALIZAÇÃO (se disponível)
    if 'location' in df.columns:
        print("✅ Processando features de 'location'")
        le_location = LabelEncoder()
        df['location_id'] = le_location.fit_transform(df['location'].astype(str))
        
        # Estatísticas por localização
        location_stats = df.groupby('location')['status'].agg(['mean', 'count']).reset_index()
        location_stats.columns = ['location', 'location_occupancy_rate', 'location_frequency']
        df = df.merge(location_stats, on='location', how='left')
        print("✅ Estatísticas por localização criadas")
    
    # 5. FEATURES DE ENTITY TYPE (se disponível)
    if 'entityType' in df.columns:
        print("✅ Processando features de 'entityType'")
        le_entity = LabelEncoder()
        df['entity_type_id'] = le_entity.fit_transform(df['entityType'].astype(str))
    
    # 6. FEATURES TEMPORAIS AVANÇADAS
    if 'timestamp_seconds' in df.columns:
        df['time_normalized'] = (df['timestamp_seconds'] - df['timestamp_seconds'].min()) / \
                               (df['timestamp_seconds'].max() - df['timestamp_seconds'].min())
        print("✅ Feature 'time_normalized' criada")
    
    # 7. FEATURES DE SEMANA (se disponível)
    if 'semana' in df.columns:
        print("✅ Processando features de 'semana'")
        df['week_number'] = pd.to_numeric(df['semana'], errors='coerce')
        df['week_number'].fillna(0, inplace=True)  # opcional: substitui NaNs por zero
        df['week_sin'] = np.sin(2 * np.pi * df['week_number'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week_number'] / 52)
        print(f"Valores únicos em 'week_number': {df['week_number'].unique()}")

    
    # 8. FEATURES DE SEQUÊNCIA (ordenar por device/location e datetime)
    # Usar a coluna de agrupamento que estiver disponível
    group_col = None
    if 'refDevice' in df.columns:
        group_col = 'refDevice'
    elif 'location' in df.columns:
        group_col = 'location'
    
    if group_col:
        print(f"✅ Criando features sequenciais agrupadas por '{group_col}'")
        df = df.sort_values([group_col, 'datetime'])
        df['prev_status'] = df.groupby(group_col)['status'].shift(1)
        df['prev_status'] = df['prev_status'].fillna(df['status'])
        df['status_changed'] = (df['status'] != df['prev_status']).astype(int)
        
        # Features de contexto temporal (janelas móveis)
        for window in [3, 5, 10]:
            if len(df) > window:
                df[f'avg_occupancy_{window}'] = df.groupby(group_col)['status'].rolling(
                    window=window, min_periods=1).mean().reset_index(0, drop=True)
        print("✅ Features sequenciais e de janela móvel criadas")
    
    return df

# Aplicar feature engineering
print("🔧 Aplicando feature engineering...")
try:
    df = create_comprehensive_features(df)
    print(f"✅ Feature engineering concluído. Shape do dataset: {df.shape}")
except Exception as e:
    print(f"❌ Erro no feature engineering: {e}")
    raise

# ============================== DATA PREPARATION ==============================
# Preparar target
if 'status' in df.columns:
    # Se status é string, converter para numérico
    if df['status'].dtype == 'object':
        unique_status = df['status'].unique()
        print(f"Status únicos encontrados: {unique_status}")
        
        # Mapear strings para números
        status_mapping = {}
        if any(s in str(unique_status).lower() for s in ['free', 'occupied']):
            status_mapping = {'free': 0, 'occupied': 1}
        elif any(s in str(unique_status).lower() for s in ['0', '1']):
            status_mapping = {'0': 0, '1': 1}
        else:
            # Criar mapeamento automático
            unique_vals = sorted(df['status'].unique())
            status_mapping = {val: i for i, val in enumerate(unique_vals)}
        
        print(f"Mapeamento de status: {status_mapping}")
        df['status'] = df['status'].map(status_mapping)
    
    y = df['status'].astype(int).values
else:
    raise ValueError("Coluna 'status' não encontrada!")

# Selecionar features para o modelo
feature_columns = [
    'hour', 'minute', 'weekday', 'day_of_month', 'month',
    'is_weekend', 'is_business_hours', 'is_rush_hour', 'is_lunch_time', 
    'is_night', 'is_morning', 'is_afternoon', 'is_evening',
    'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 
    'month_sin', 'month_cos'
]

# Adicionar features condicionalmente se existirem
conditional_features = [
    'device_id', 'location_id', 'entity_type_id', 'week_number',
    'week_sin', 'week_cos', 'time_normalized', 'prev_status', 'status_changed',
    'location_occupancy_rate', 'location_frequency'
]

for feature in conditional_features:
    if feature in df.columns:
        feature_columns.append(feature)

# Adicionar features de dispositivo one-hot se existirem
device_cols = [col for col in df.columns if col.startswith('device_')]
feature_columns.extend(device_cols)

# Adicionar features de média móvel se existirem
avg_cols = [col for col in df.columns if col.startswith('avg_occupancy_')]
feature_columns.extend(avg_cols)

# Filtrar apenas features que existem no DataFrame
feature_columns = [col for col in feature_columns if col in df.columns]

print(f"📊 Features selecionadas ({len(feature_columns)}): {feature_columns[:10]}...")
if len(feature_columns) > 10:
    print(f"    ... e mais {len(feature_columns) - 10} features")

# Preparar matriz de features
X = df[feature_columns].copy()

# Handle any remaining NaN values
X = X.fillna(0)

# Convert to numpy array
X = X.values.astype(np.float32)

# Normalização
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"Shape dos dados: X={X.shape}, y={y.shape}")
print(f"Classes únicas: {np.unique(y)}")
print(f"Distribuição de classes: {np.bincount(y)}")

# Verificar balanceamento
class_counts = np.bincount(y)
if len(class_counts) >= 2:
    minority_ratio = min(class_counts) / sum(class_counts)
    print(f"Ratio da classe minoritária: {minority_ratio:.3f}")

# ============================== TRAIN-TEST SPLIT ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Dados de treino: {len(X_train)}, Dados de teste: {len(X_test)}")

# ============================== BUILD IMPROVED MODEL ==============================
def build_improved_model(input_dim):
    """Construir modelo melhorado com arquitetura otimizada"""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        
        # Primeira camada com mais neurônios
        keras.layers.Dense(128, activation="relu", kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        # Segunda camada
        keras.layers.Dense(64, activation="relu", kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.25),
        
        # Terceira camada
        keras.layers.Dense(32, activation="relu", kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        
        # Camada de saída
        keras.layers.Dense(1, activation="sigmoid")
    ])
    
    # Optimizer com learning rate personalizado
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", "precision", "recall"]
    )
    
    return model

model = build_improved_model(X_train.shape[1])
print("✅ Modelo melhorado criado")
print(f"Parâmetros do modelo: {model.count_params()}")

# ============================== CLASS WEIGHTS ==============================
if len(np.unique(y_train)) > 1:
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    cw_dict = dict(enumerate(class_weights))
    print(f"Class weights aplicados: {cw_dict}")
else:
    cw_dict = None

# ============================== CALLBACKS ==============================
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# ============================== TRAIN ==============================
print("🏋️ Iniciando treinamento...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=cw_dict,
    callbacks=callbacks,
    verbose=1
)

# ============================== EVALUATE ==============================
print("🧪 Avaliando modelo...")
loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

print(f"\n✅ RESULTADOS FINAIS:")
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")

# Predições detalhadas
y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)
print(f"\n📊 RELATÓRIO DE CLASSIFICAÇÃO:")
target_names = ['Free', 'Occupied'] if len(np.unique(y)) == 2 else [f'Class_{i}' for i in np.unique(y)]
print(classification_report(y_test, y_pred, target_names=target_names))

# ============================== VISUALIZATION ==============================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Accuracy
axes[0,0].plot(history.history['accuracy'], label='Train Accuracy', marker='o')
axes[0,0].plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')
axes[0,0].set_title('Model Accuracy')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Accuracy')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. Loss
axes[0,1].plot(history.history['loss'], label='Train Loss', marker='o')
axes[0,1].plot(history.history['val_loss'], label='Val Loss', marker='s')
axes[0,1].set_title('Model Loss')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Loss')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 3. Precision
axes[0,2].plot(history.history['precision'], label='Train Precision', marker='o')
axes[0,2].plot(history.history['val_precision'], label='Val Precision', marker='s')
axes[0,2].set_title('Model Precision')
axes[0,2].set_xlabel('Epoch')
axes[0,2].set_ylabel('Precision')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

# 4. Recall
axes[1,0].plot(history.history['recall'], label='Train Recall', marker='o')
axes[1,0].plot(history.history['val_recall'], label='Val Recall', marker='s')
axes[1,0].set_title('Model Recall')
axes[1,0].set_xlabel('Epoch')
axes[1,0].set_ylabel('Recall')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# 5. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,1])
axes[1,1].set_title('Confusion Matrix')
axes[1,1].set_xlabel('Predicted')
axes[1,1].set_ylabel('Actual')

# 6. Feature Importance (aproximação usando pesos da primeira camada)
if len(feature_columns) <= 20:  # Só mostrar se não for muitas features
    weights = model.layers[0].get_weights()[0]
    feature_importance = np.mean(np.abs(weights), axis=1)
    
    # Top 10 features mais importantes
    top_indices = np.argsort(feature_importance)[-10:]
    top_features = [feature_columns[i] for i in top_indices]
    top_importance = feature_importance[top_indices]
    
    axes[1,2].barh(range(len(top_features)), top_importance)
    axes[1,2].set_yticks(range(len(top_features)))
    axes[1,2].set_yticklabels(top_features)
    axes[1,2].set_title('Top 10 Feature Importance')
    axes[1,2].set_xlabel('Importance')
else:
    axes[1,2].text(0.5, 0.5, f'Modelo usa {len(feature_columns)} features\n(muito para visualizar)', 
                   ha='center', va='center', transform=axes[1,2].transAxes)
    axes[1,2].set_title('Feature Count')

plt.tight_layout()
plt.savefig("improved_centralized_training_performance.png", dpi=150, bbox_inches='tight')
print("📈 Gráfico salvo: improved_centralized_training_performance.png")
plt.show()

# ============================== SAVE MODEL ==============================
MODEL_DIR = "smartparknet_dashboard/backend/modelos_centralized"
os.makedirs(MODEL_DIR, exist_ok=True)

# Salvar modelo
model.save(os.path.join(MODEL_DIR, "model_centralized_improved.h5"))

# Salvar scaler
import pickle
with open(os.path.join(MODEL_DIR, "scaler_centralized.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# Salvar lista de features
with open(os.path.join(MODEL_DIR, "features_centralized.txt"), "w") as f:
    for feature in feature_columns:
        f.write(f"{feature}\n")

print(f"💾 Modelo, scaler e features salvos em: {MODEL_DIR}/")
print(f"💾 Arquivos criados:")
print(f"  - model_centralized_improved.h5")
print(f"  - scaler_centralized.pkl") 
print(f"  - features_centralized.txt")

print(f"\n🎯 FEATURES UTILIZADAS:")
print(f"Total de features: {len(feature_columns)}")
for i, feature in enumerate(feature_columns, 1):
    print(f"  {i:2d}. {feature}")

print(f"\n🚀 TREINAMENTO CONCLUÍDO COM SUCESSO!")