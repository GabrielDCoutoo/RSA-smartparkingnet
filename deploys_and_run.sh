#!/bin/bash

# ================================
# CONFIGURAÇÃO
# ================================
PASSWORD="openlab"
USERNAME="nap"
SCRIPT_PATH="RSA/cmd/clientML.py"
MODEL_PATH="RSA/smartparknet_dashboard/backend/modelos_federated/model_global.h5"
REMOTE_MODEL_DIR="RSA/model_global"
REMOTE_SCRIPT_PATH="RSA/cmd/clientML.py"

# Lista de IPs dos clientes
CLIENTS=("192.168.3.16" "192.168.3.19" "192.168.3.6")
# Caminhos remotos dos CSVs (um por cliente)
DATASETS=("RSA/smartparknet_dashboard/backend/camera_022.  csv" "RSA/smartparknet_dashboard/backend/camera_033.csv" "RSA/smartparknet_dashboard/backend/camera_035.csv")

# ================================
# ENVIO E EXECUÇÃO REMOTA
# ================================
for i in "${!CLIENTS[@]}"; do
    IP="${CLIENTS[$i]}"
    DATASET_PATH="${DATASETS[$i]}"

    echo "🚀 A enviar modelo global para $IP..."
    sshpass -p "$PASSWORD" ssh "$USERNAME@$IP" "mkdir -p $REMOTE_MODEL_DIR"
    sshpass -p "$PASSWORD" scp "$MODEL_PATH" "$USERNAME@$IP:$REMOTE_MODEL_DIR/model_global.h5"

    echo "📂 A lançar treino remoto em $IP com dataset $DATASET_PATH..."
    sshpass -p "$PASSWORD" ssh "$USERNAME@$IP" "
        export DATASET=$DATASET_PATH &&
        export EPOCHS=5 &&
        python3 $REMOTE_SCRIPT_PATH
    " || echo "❌ Erro ao treinar em $IP"
done

echo "✅ Execução concluída."
