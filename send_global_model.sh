#!/bin/bash

# 🔐 Configurações
USERNAME="nap"
PASSWORD="openlab"
CLIENTS=("192.168.3.16" "192.168.3.19" "192.168.3.6")

# 📁 Caminho local do modelo global
MODEL_PATH="modelos_federated/model_global.h5"

# 📁 Caminho remoto onde os clientes vão guardar o modelo global
REMOTE_DIR="RSA/model_global"

# Enviar o modelo para cada cliente
for CLIENT in "${CLIENTS[@]}"; do
    echo "📤 A enviar modelo para $CLIENT..."
    sshpass -p "$PASSWORD" ssh "$USERNAME@$CLIENT" "mkdir -p $REMOTE_DIR"
    sshpass -p "$PASSWORD" scp "$MODEL_PATH" "$USERNAME@$CLIENT:$REMOTE_DIR/" \
        && echo "✅ Modelo enviado com sucesso para $CLIENT" \
        || echo "❌ Falha ao enviar modelo para $CLIENT"
done

echo "📦 Modelo global distribuído para todos os clientes."
