import os
import pandas as pd

# Caminho do ficheiro a usar como base
input_csv = "smartparknet_dashboard/backend/forecast_b/bbb/dados_Câmara_035_bbb_6.csv"
output_csv = "datasets/teste_comum.csv"

# Criar a pasta se não existir
os.makedirs("datasets", exist_ok=True)

# Carregar, garantir que só há a coluna 'status', e guardar
df = pd.read_csv(input_csv)

# Apenas manter colunas essenciais
df = df[["recvTime", "status"]]
df.to_csv(output_csv, index=False)

print(f"✅ Ficheiro de teste comum gerado em: {output_csv}")
