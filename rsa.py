import pandas as pd
import os

# --- CONFIGURAÇÃO ---
CAMARAS = {
    "urn:ngsi-ld:SlpCamera:033-1": "Câmara_033",
    "urn:ngsi-ld:SlpCamera:022-1": "Câmara_022",
    "urn:ngsi-ld:SlpCamera:035-1": "Câmara_035"
}
INTERVALO_INICIO = "2025-02-26"
INTERVALO_FIM = "2025-04-26"
DADOS_PATH = "Fev-Apr-smarparking.csv"
OUTPUT_DIR = "smartparknet_dashboard/backend"
FORECAST_DIR = os.path.join(OUTPUT_DIR, "forecast_b")
MIN_AMOSTRAS_POR_CSV = 50
LOG_PATH = os.path.join(OUTPUT_DIR, "resumo.log")

# --- LEITURA E FILTRAGEM INICIAL ---
df = pd.read_csv(DADOS_PATH, low_memory=False)
df['recvTime'] = pd.to_datetime(df['recvTime'], errors='coerce')
df = df.dropna(subset=['recvTime'])

df = df[
    (df['recvTime'] >= INTERVALO_INICIO) &
    (df['recvTime'] <= INTERVALO_FIM) &
    (df['refDevice'].isin(CAMARAS.keys())) &
    (~df['entityId'].str.endswith(":0000"))
]

df['refDevice'] = df['refDevice'].map(CAMARAS)
df['status'] = df['status'].map({'free': 0, 'occupied': 1})
df = df.sort_values(by=['refDevice', 'entityId', 'recvTime'])

# --- REMOVER TRANSIÇÕES INSTÁVEIS ---
def marcar_transicoes_instaveis(grupo):
    grupo = grupo.copy()
    grupo['next_status'] = grupo['status'].shift(-1)
    grupo['next_time'] = grupo['recvTime'].shift(-1)
    grupo['tempo_prox'] = (grupo['next_time'] - grupo['recvTime']).dt.total_seconds()
    grupo['mudanca_instavel'] = (
        (grupo['status'] != grupo['next_status']) &
        (grupo['tempo_prox'] <= 600)
    )
    return grupo

df = df.groupby('entityId', group_keys=False).apply(marcar_transicoes_instaveis, include_groups=False)
df = df[~df['mudanca_instavel']].copy()
df.drop(columns=['next_status', 'next_time', 'tempo_prox', 'mudanca_instavel'], inplace=True)

# --- EXPORTAÇÃO BASE ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
df.to_csv(os.path.join(OUTPUT_DIR, "dados_filtrados_por_camara.csv"), index=False)

for camara in df['refDevice'].unique():
    df[df['refDevice'] == camara].to_csv(
        os.path.join(OUTPUT_DIR, f"dados_{camara}.csv"),
        index=False
    )

# --- EXPORTAÇÃO SEMANAL ---
df['semana'] = df['recvTime'].dt.to_period('W-MON')
semanas_ordenadas = sorted(df['semana'].unique())

for tipo in ['b', 'bb', 'bbb','bbbb']:
    os.makedirs(os.path.join(FORECAST_DIR, tipo), exist_ok=True)

with open(LOG_PATH, "w") as log:
    log.write("📊 RESUMO DE EXPORTAÇÃO POR CÂMARA E SEMANA\n")
    log.write("=" * 60 + "\n\n")

    for camara in df['refDevice'].unique():
        df_camara = df[df['refDevice'] == camara]
        log.write(f"▶️ {camara} — {len(df_camara)} registos filtrados no total\n")

        for i in range(2, len(semanas_ordenadas)):
            semana_n = semanas_ordenadas[i]
            label = f"semana {i} ({semana_n.start_time.date()} - {semana_n.end_time.date()})"
            log.write(f"  🗓️ {label}\n")

            def exportar_csv(df_input, nome_ficheiro, tipo):
                df_input = df_input.copy()
                if len(df_input) >= MIN_AMOSTRAS_POR_CSV:
                    # 🔁 Limita a 15.000 amostras de forma aleatória (mas ordenadas por tempo)
                    if len(df_input) > 15000:
                        df_input = df_input.sample(n=15000, random_state=42).sort_values(by="recvTime")

                    inicio = df_input['recvTime'].min()
                    df_input.loc[:, 'timestamp_seconds'] = (df_input['recvTime'] - inicio).dt.total_seconds()
                    df_input[['timestamp_seconds', 'status']].to_csv(nome_ficheiro, index=False)
                    log.write(f"     ✔️ {tipo}.csv — {len(df_input)} amostras\n")
                else:
                    log.write(f"     ❌ {tipo}.csv não gerado (apenas {len(df_input)} amostras)\n")


            # b
            df_b = df_camara[df_camara['semana'] == semana_n]
            exportar_csv(df_b, f"{FORECAST_DIR}/b/dados_{camara}_b_{i}.csv", 'b')

            # bb
            df_bb = df_camara[df_camara['semana'].isin([semanas_ordenadas[i-1], semana_n])]
            exportar_csv(df_bb, f"{FORECAST_DIR}/bb/dados_{camara}_bb_{i}.csv", 'bb')

            # bbb
            df_bbb = df_camara[df_camara['semana'].isin([semanas_ordenadas[i-2], semanas_ordenadas[i-1], semana_n])]
            exportar_csv(df_bbb, f"{FORECAST_DIR}/bbb/dados_{camara}_bbb_{i}.csv", 'bbb')

            # bbbb
            df_bbbb = df_camara[df_camara['semana'].isin([
                semanas_ordenadas[i-3], semanas_ordenadas[i-2],
                semanas_ordenadas[i-1], semana_n
            ])]
            exportar_csv(df_bbbb, f"{FORECAST_DIR}/bbbb/dados_{camara}_bbbb_{i}.csv", 'bbbb')
        log.write("\n")

print("✅ Script executado com sucesso e sem warnings!")
