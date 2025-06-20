import tensorflow as tf
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from glob import glob

# ====================== AGREGAÇÃO DOS MODELOS ======================

# Pasta onde estão os modelos .h5 dos clientes
MODEL_DIR = "histories"
OUTPUT_PATH = "modelos_federated/model_global.h5"

# Encontrar todos os ficheiros .h5 (pesos dos modelos locais)
model_paths = sorted(glob(os.path.join(MODEL_DIR, "client_*.h5")))
print(f"📂 Modelos encontrados: {model_paths}")

# Carregar os modelos e extrair os pesos
models = [tf.keras.models.load_model(path) for path in model_paths]
weights = [model.get_weights() for model in models]

# Agregação média (FedAvg)
avg_weights = []
for layers in zip(*weights):
    avg_layer = np.mean(np.array(layers), axis=0)
    avg_weights.append(avg_layer)

# Criar modelo global com a mesma arquitetura
global_model = tf.keras.models.clone_model(models[0])
global_model.set_weights(avg_weights)

# Guardar modelo global
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
global_model.save(OUTPUT_PATH)
print(f"✅ Modelo global guardado em {OUTPUT_PATH}")

