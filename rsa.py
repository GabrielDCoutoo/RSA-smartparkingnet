import pandas as pd
import os

# Lê o ficheiro 
df = pd.read_csv("Fev-Apr-smarparking.csv", low_memory=False)

# Converte a coluna de tempo
df['recvTime'] = pd.to_datetime(df['recvTime'], errors='coerce')
df = df.dropna(subset=['recvTime'])

# Define o intervalo
inicio = "2025-02-26"
fim = "2025-03-04"

# Mapeamento para nomes legíveis das câmaras
mapa_nomes = {
    "urn:ngsi-ld:SlpCamera:033-1": "Câmara 033",
    "urn:ngsi-ld:SlpCamera:022-1": "Câmara 022",
    "urn:ngsi-ld:SlpCamera:035-1": "Câmara 035"
}

# Filtra pelo intervalo de tempo e pelas câmaras desejadas
camaras_validas = list(mapa_nomes.keys())

# Remover lugares inválidos
df = df[
    (df['recvTime'] >= inicio) &
    (df['recvTime'] <= fim) &
    (df['refDevice'].isin(camaras_validas)) &
    (~df['entityId'].str.endswith(":0000"))
]

# Substitui os nomes longos por nomes curtos legíveis
df['refDevice'] = df['refDevice'].map(mapa_nomes)

# Ordena os dados
df = df.sort_values(by=['entityId', 'recvTime'])

# Função para remover mudanças repentinas
def marcar_transicoes_instaveis(grupo, intervalo_segundos=60):
    grupo['next_status'] = grupo['status'].shift(-1)
    grupo['next_time'] = grupo['recvTime'].shift(-1)
    grupo['tempo_prox'] = (grupo['next_time'] - grupo['recvTime']).dt.total_seconds()

    grupo['mudanca_instavel'] = (
        (grupo['status'] != grupo['next_status']) &
        (grupo['tempo_prox'] <= intervalo_segundos)
    )
    return grupo

# Aplica por lugar e limpa
df = df.groupby('entityId', group_keys=False).apply(marcar_transicoes_instaveis)

# Exporta apenas instáveis para inspeção
df[df['mudanca_instavel'] == True].to_csv("mudancas_instaveis.csv", index=False)

# Cria df limpo
df_limpo = df[~df['mudanca_instavel']].copy()
df_limpo = df_limpo.drop(columns=['next_status', 'next_time', 'tempo_prox', 'mudanca_instavel'])
df_limpo = df_limpo.sort_values(by=['refDevice', 'recvTime'])

# Garante que a pasta de destino existe
output_dir = "smartparknet_dashboard/backend"
os.makedirs(output_dir, exist_ok=True)

# Exporta o CSV final com dados válidos de todas as câmaras
output_path = os.path.join(output_dir, "dados_filtrados_por_camara.csv")
df_limpo.to_csv(output_path, index=False)

print(" CSVs gerados:")
print("- mudanças_instaveis.csv")
print(f"- {output_path}")
