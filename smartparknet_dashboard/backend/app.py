from flask import Flask, jsonify
import pandas as pd
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route("/api/camaras")
def get_ocupacao_camaras():
    # Caminho absoluto para garantir leitura correta do CSV
    base_dir = os.path.abspath(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, 'dados_filtrados_por_camara.csv')
    
    df = pd.read_csv(csv_path)

    # Garantir que 'recvTime' é datetime e válido
    df['recvTime'] = pd.to_datetime(df['recvTime'], errors='coerce')
    df = df.dropna(subset=['recvTime'])

    # Extrair hora e dia
    df['hora'] = df['recvTime'].dt.strftime('%H:00')
    df['dia'] = df['recvTime'].dt.day_name()

    # Agrupar por câmara, dia, hora e status (0 = livre, 1 = ocupado)
    grouped = df.groupby(['refDevice', 'dia', 'hora', 'status']).size().unstack(fill_value=0).reset_index()

    resposta = []
    for camara in grouped['refDevice'].unique():
        dados_camara = grouped[grouped['refDevice'] == camara]
        dados = []
        for _, row in dados_camara.iterrows():
            ocupado = int(row.get(1, 0))  # status == 1
            livre = int(row.get(0, 0))    # status == 0
            dados.append({
                "tempo": f"{row['dia']} {row['hora']}",
                "ocupado": ocupado,
                "livre": livre
            })
        resposta.append({
            "camara": camara,
            "dados": dados
        })

    return jsonify(resposta)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
