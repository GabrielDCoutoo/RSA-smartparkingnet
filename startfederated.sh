#!/bin/bash

MODE=$1  # centralized ou federated
CENTRALIZED_DATASET="smartparknet_dashboard/backend/dados_centralizado.csv"
CENTRALIZED_SCRIPT="centralized_train.py"

echo "🚀 Sistema de aprendizagem (${MODE:-federated})"

if [ "$MODE" == "centralized" ]; then
    echo "🔍 A iniciar treino CENTRALIZADO..."
    if [ ! -f "$CENTRALIZED_DATASET" ]; then
        echo "❌ Dataset centralizado não encontrado em $CENTRALIZED_DATASET!"
        exit 1
    fi

    if [ ! -f "$CENTRALIZED_SCRIPT" ]; then
        echo "❌ Script $CENTRALIZED_SCRIPT não encontrado!"
        exit 1
    fi

    python3 "$CENTRALIZED_SCRIPT"
    exit 0
fi

# 🚀 Se não for centralizado, inicia modo federado:
echo "🚀 A iniciar sistema de aprendizagem FEDERADA..."

PASTAS=("b" "bb" "bbb" "bbbb")
BASE_DIR="smartparknet_dashboard/backend/forecast_b"
SERVER_SCRIPT="server/serverflower.py"
CLIENT_SCRIPT="cmd/clientML.py"

# Verificar diretórios
if [ ! -d "$BASE_DIR" ]; then
    echo "❌ Diretório $BASE_DIR não encontrado!"
    echo "Execute primeiro o script rsa.py para gerar os dados"
    exit 1
fi

# Contar CSVs
total_csvs=0
csv_files=()
for pasta in "${PASTAS[@]}"; do
    if [ -d "$BASE_DIR/$pasta" ]; then
        count=$(find "$BASE_DIR/$pasta" -name "*.csv" -type f | wc -l)
        echo "📁 $pasta: $count CSVs encontrados"
        for csv in "$BASE_DIR/$pasta"/*.csv; do
            [[ -f "$csv" ]] && csv_files+=("$csv") && ((total_csvs++))
        done
    fi
done

if [ $total_csvs -eq 0 ]; then
    echo "❌ Nenhum ficheiro CSV encontrado!"
    exit 1
fi

MAX_CLIENTS=$total_csvs
echo "📊 Total datasets: $total_csvs"
echo "🔢 Máximo de clientes: $MAX_CLIENTS"
echo ""

# Limpeza
cleanup() {
    echo "🛑 A terminar processos..."
    pkill -f "python3.*serverflower.py" 2>/dev/null
    pkill -f "python3.*clientML.py" 2>/dev/null
    rm -f server_port.txt 2>/dev/null
    echo "✅ Limpeza concluída"
    exit 0
}

trap cleanup SIGINT SIGTERM

if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 não encontrado!"
    exit 1
fi

rm -f server_port.txt

echo "🖥️ A iniciar servidor Flower..."
python3 "$SERVER_SCRIPT" &
SERVER_PID=$!

echo "⏳ A aguardar servidor inicializar..."
for i in {1..15}; do
    if [ -f "server_port.txt" ]; then
        SERVER_PORT=$(cat server_port.txt)
        echo "✅ Servidor na porta $SERVER_PORT (PID: $SERVER_PID)"
        break
    fi
    sleep 1
done

if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Servidor falhou!"
    cleanup
fi

cliente_num=1
client_pids=()

for csv in "${csv_files[@]:0:$MAX_CLIENTS}"; do
    echo "📡 Cliente $cliente_num: $(basename "$csv")"

    export DATASET="$csv"
    export PLOT_PERFORMANCE="false"
    export EPOCHS="5"
    export BATCH_SIZE="16"

    python3 "$CLIENT_SCRIPT" &
    CLIENT_PID=$!
    client_pids+=($CLIENT_PID)

    echo "   └── PID: $CLIENT_PID"
    sleep 5
    ((cliente_num++))

    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "❌ Servidor parou!"
        break
    fi
done

echo "🎯 Clientes iniciados: ${#client_pids[@]}"
echo "📊 Monitorizando progresso..."

while kill -0 $SERVER_PID 2>/dev/null; do
    active_clients=0
    for pid in "${client_pids[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            ((active_clients++))
        fi
    done

    if [ $active_clients -eq 0 ]; then
        echo "✅ Todos os clientes terminaram"
        break
    fi

    echo "🔄 Clientes ativos: $active_clients"
    sleep 10
done

if kill -0 $SERVER_PID 2>/dev/null; then
    echo "🛑 A terminar servidor..."
    kill $SERVER_PID
    sleep 2
fi

cleanup
echo "✅ Aprendizagem federada concluída!"
