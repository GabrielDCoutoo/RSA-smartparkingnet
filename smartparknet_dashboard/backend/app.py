from flask import Flask, jsonify
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/api/camaras")
def get_ocupacao_camaras():
    df = pd.read_csv("camaras_filtradas.csv", parse_dates=["recvTime"])

    df['hora'] = df['recvTime'].dt.strftime('%H:00')
    
    # Agrupa por câmara e hora, contando estados
    grouped = df.groupby(['refDevice', 'hora', 'status']).size().unstack(fill_value=0).reset_index()

    # Prepara o formato para o frontend
    resposta = []
    for camara in grouped['refDevice'].unique():
        dados_camara = grouped[grouped['refDevice'] == camara]
        dados = []
        for _, row in dados_camara.iterrows():
            dados.append({
                "tempo": row['hora'],
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
