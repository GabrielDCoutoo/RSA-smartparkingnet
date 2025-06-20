#!/bin/bash

# 📁 Pasta local onde queres guardar os históricos e modelos
DEST_DIR="./histories"

# 🖥️ IPs ou hostnames das Raspberrys
CLIENTS=("192.168.3.16" "192.168.3.19" "192.168.3.6")

# 🔐 Nome do utilizador nas Raspberrys
USERNAME="nap"
PASSWORD="openlab"

# Caminhos remotos a copiar (ficheiros .json e .h5)
REMOTE_JSON_PATH="~/RSA/history_d*.json"
REMOTE_H5_PATH="~/RSA/smartparknet_dashboard/backend/modelos_federated/client_*.h5"

# Criar pasta local se não existir
mkdir -p "$DEST_DIR"

echo "🚀 A copiar históricos (.json) e modelos (.h5) dos clientes para $DEST_DIR"

for CLIENT in "${CLIENTS[@]}"; do
    echo "📡 A copiar de $CLIENT..."

    # Copiar históricos
    sshpass -p "$PASSWORD" scp ${USERNAME}@${CLIENT}:"$REMOTE_JSON_PATH" "$DEST_DIR/" \
        && echo "✅ Históricos copiados" \
        || echo "❌ Falha ao copiar históricos de $CLIENT"

    # Copiar modelos
    sshpass -p "$PASSWORD" scp ${USERNAME}@${CLIENT}:"$REMOTE_H5_PATH" "$DEST_DIR/" \
        && echo "✅ Modelos copiados" \
        || echo "❌ Falha ao copiar modelos de $CLIENT"
done

echo "📊 Todos os ficheiros foram copiados para $DEST_DIR"
