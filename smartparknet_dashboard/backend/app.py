from flask import Flask, jsonify
import pandas as pd
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route("/api/camaras")
def get_ocupacao_camaras():
    csv_path = os.path.join(os.path.dirname(__file__), 'dados_filtrados_por_camara.csv')
    df = pd.read_csv(csv_path)

    df = df[df['recvTime'].apply(lambda x: isinstance(x, str) and len(x) > 10)]
    df['recvTime'] = pd.to_datetime(df['recvTime'], errors='coerce')
    df = df.dropna(subset=['recvTime'])

    df['hora'] = df['recvTime'].dt.strftime('%H:00')
    df['dia'] = df['recvTime'].dt.day_name()

    grouped = df.groupby(['refDevice', 'dia', 'hora', 'status']).size().unstack(fill_value=0).reset_index()

    resposta = []
    for camara in grouped['refDevice'].unique():
        dados_camara = grouped[grouped['refDevice'] == camara]
        dados = []
        for _, row in dados_camara.iterrows():
            dados.append({
                "tempo": f"{row['dia']} {row['hora']}",
                "ocupado": int(row.get('occupied', 0)),
                "livre": int(row.get('free', 0))
            })
        resposta.append({
            "camara": camara,
            "dados": dados
        })

    return jsonify(resposta)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
