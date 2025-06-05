import os
import subprocess
import sys

# Caminho base onde estão os CSVs divididos por b, bb, bbb
pastas = [
    "smartparknet_dashboard/backend/forecast_b/b",
    "smartparknet_dashboard/backend/forecast_b/bb",
    "smartparknet_dashboard/backend/forecast_b/bbb"
]

# Caminho para o script de treino
script_treino = "train_LSTM.py"
summary_lines = []

for pasta in pastas:
    print(f"\n📂 Pasta: {pasta}")
    for ficheiro in os.listdir(pasta):
        if ficheiro.endswith(".csv"):
            caminho_csv = os.path.join(pasta, ficheiro)

            # Extrair info do nome do ficheiro
            partes = ficheiro.replace(".csv", "").split("_")
            camara = "_".join(partes[1:3])  # ex: Câmara_033
            tipo = partes[3]
            semana = partes[4]

            modelo_dir = os.path.join("smartparknet_dashboard/backend/modelos", camara, tipo)
            modelo_path = os.path.join(modelo_dir, f"modelo_{tipo}_{semana}.h5")
            log_path = os.path.join(modelo_dir, f"log_{tipo}_{semana}.txt")
            relatorio_path = os.path.join(modelo_dir, f"relatorio_{tipo}_{semana}.txt")

            os.makedirs(modelo_dir, exist_ok=True)

            if os.path.exists(modelo_path):
                print(f"✅ Já treinado: {ficheiro}")
                summary_lines.append(f"✔️ {ficheiro} — já existia")
                continue

            print(f"🚀 A treinar modelo para: {ficheiro}")
            with open(log_path, "w") as log_file:
                result = subprocess.run(
                    [sys.executable, script_treino, caminho_csv],
                    stdout=log_file,
                    stderr=log_file
                )

            if os.path.exists(modelo_path):
                summary_lines.append(f"✔️ {ficheiro} — modelo: {modelo_path}, relatório: {relatorio_path}")
            else:
                summary_lines.append(f"❌ {ficheiro} — falhou (ver log em {log_path})")
            print(f"📝 Log guardado em {log_path}")

# Gravar o resumo final
summary_path = "smartparknet_dashboard/backend/modelos/summary.log"
with open(summary_path, "w") as summary_file:
    summary_file.write("📋 RESUMO FINAL DE TREINO\n")
    summary_file.write("=" * 60 + "\n\n")
    summary_file.write("\n".join(summary_lines))

print(f"\n📄 Summary guardado em {summary_path}")
