import pandas as pd

# Lê o ficheiro grande
df = pd.read_csv("Fev-Apr-smarparking.csv", low_memory=False)

# Converte a coluna de tempo
df['recvTime'] = pd.to_datetime(df['recvTime'])

# Define o intervalo alargado
inicio = "2025-02-26"
fim = "2025-03-04"

# Filtra pelo intervalo de tempo e câmara
df_filtrado = df[
    (df['recvTime'] >= inicio) &
    (df['recvTime'] <= fim) &
    (df['refDevice'] == "urn:ngsi-ld:SlpCamera:033-1")
]

#  ParkingSpots com ID 0000 (inválidos)
df_filtrado = df_filtrado[~df_filtrado['entityId'].str.endswith(":0000")]

# Ordena os dados
df_filtrado = df_filtrado.sort_values(by=['entityId', 'recvTime'])

# Função para remover mudanças de estado muito rápidas (ex: < 30s)
def remover_mudancas_repentinas(grupo, intervalo_segundos=30):
    grupo['prev_status'] = grupo['status'].shift(1)
    grupo['prev_time'] = grupo['recvTime'].shift(1)
    grupo['tempo_dif'] = (grupo['recvTime'] - grupo['prev_time']).dt.total_seconds()

    limpo = grupo[
        (grupo['status'] == grupo['prev_status']) |
        (grupo['tempo_dif'] > intervalo_segundos) |
        (grupo['prev_status'].isna())
    ]

    return limpo.drop(columns=['prev_status', 'prev_time', 'tempo_dif'])

# Aplica a limpeza por ParkingSpot
df_limpo = df_filtrado.groupby('entityId', group_keys=False).apply(remover_mudancas_repentinas)

# Ordena final por câmara e tempo
df_camaras = df_limpo.sort_values(by=['refDevice', 'recvTime'])

# Guarda em CSV
df_camaras.to_csv("dados_filtrados_por_camara.csv", index=False)

print("✅ Ficheiro 'dados_filtrados_por_camara.csv' gerado com sucesso (sem spots 0000)!")
