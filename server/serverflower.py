import flwr as fl
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from flwr.common import parameters_to_ndarrays

# Função para criar modelo idêntico ao do cliente
def build_model(input_shape):
    model = Sequential([
        Dense(32, activation="relu", input_shape=(input_shape,)),
        Dense(32, activation="relu"),
        Dense(2, activation="softmax")  # Para classificação binária (0 e 1)
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Modelo inicial (será atualizado quando o primeiro cliente se conectar)
model = None

# Função de avaliação
def evaluate(server_round, parameters, config):
    global model
    if model is None:
        # Criar modelo com shape padrão se ainda não existir
        model = build_model(1)  # Será ajustado quando receber os primeiros parâmetros
    
    try:
        model.set_weights(parameters)
        # Criar dados dummy para avaliação (pode substituir por dados reais se tiver)
        X_dummy = np.random.rand(10, 1)  # Ajustar conforme necessário
        y_dummy = np.random.randint(0, 2, size=(10,))
        y_dummy_cat = tf.keras.utils.to_categorical(y_dummy, 2)
        
        loss, accuracy = model.evaluate(X_dummy, y_dummy_cat, verbose=0)
        return loss, {"accuracy": accuracy}
    except Exception as e:
        print(f"Erro na avaliação: {e}")
        return 0.0, {"accuracy": 0.0}

# Estratégia personalizada para lidar com inicialização dinâmica
class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_initialized = False
    
    def initialize_parameters(self, client_manager):
        # Aguardar que um cliente forneça os parâmetros iniciais
        return None
    
    def aggregate_fit(self, server_round, results, failures):
        global model

        # Na primeira ronda, inicializar o modelo com base nos parâmetros do cliente
        if not self.model_initialized and results:
            first_params = results[0][1].parameters
            params_ndarrays = parameters_to_ndarrays(first_params)
            if len(params_ndarrays) >= 1:
                input_shape = params_ndarrays[0].shape[0]
                model = build_model(input_shape)
                self.model_initialized = True
                print(f"✅ Modelo inicializado com input_shape: {input_shape}")

        return super().aggregate_fit(server_round, results, failures)

# Configurar estratégia
strategy = CustomFedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=1,  # Mínimo 1 cliente para começar
    min_evaluate_clients=1,
    min_available_clients=1,
    evaluate_fn=evaluate
)

# Iniciar servidor
if __name__ == "__main__":
    print("🚀 A iniciar servidor Flower...")
    print("Aguardando clientes na porta 8080...")

    # Gravar porta num ficheiro para sincronização com o script bash
    with open("server_port.txt", "w") as f:
        f.write("8080")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy
    )