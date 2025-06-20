import os
import json
import matplotlib.pyplot as plt
import re

# Diretório onde estão os .json (histories)
DIR = "histories"

# Lista de ficheiros .json
files = [f for f in os.listdir(DIR) if f.endswith(".json")]
print(f"📂 Ficheiros encontrados: {files}")

# Inicializar listas
all_histories = []
labels = []

# Regex para extrair nome da câmara do nome do ficheiro
camera_regex = re.compile(r"history_dados_Câmara_(\d+)_")

for file in files:
    with open(os.path.join(DIR, file), 'r') as f:
        history = json.load(f)
        # Só adiciona se a última accuracy for > 0.78
        acc = history.get("accuracy", [])
        if acc and acc[-1] > 0.78:
            all_histories.append(history)
            # Extrair nome da câmara do ficheiro
            match = camera_regex.search(file)
            if match:
                labels.append(f"Câmara {match.group(1)}")
            else:
                labels.append(file)

# Plot dos gráficos
def plot_metric(metric):
    plt.figure(figsize=(10, 5))
    for i, hist in enumerate(all_histories):
        if metric in hist:
            plt.plot(hist[metric], label=labels[i])
    plt.title(f"{metric.capitalize()} por época (accuracy final > 78%)")
    plt.xlabel("Época")
    plt.ylabel(metric.capitalize())
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"grafico_filtrado_{metric}.png")
    plt.show()

plot_metric("accuracy")
plot_metric("val_accuracy")
plot_metric("loss")
plot_metric("val_loss")
