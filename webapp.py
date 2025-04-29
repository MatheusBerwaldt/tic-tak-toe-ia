from flask import Flask, render_template_string, request, jsonify
from datasetLoader import DatasetLoader
from modelTrainer import ModelTrainer
import pandas as pd

app = Flask(__name__)

# Carrega modelo e encoder
loader = DatasetLoader()
X_train, X_val, X_test, y_train, y_val, y_test, encoder = loader.load_data()
trainer = ModelTrainer()
trainer.train_and_evaluate(X_train, X_val, y_train, y_val)
# Seleciona o modelo de maior acurácia
best_model_name, best_model_info = max(trainer.results.items(), key=lambda x: x[1]['accuracy'])
best_model = best_model_info['model']
print(f"Modelo selecionado: {best_model_name} (Acurácia: {best_model_info['accuracy']:.2%})")

# Variáveis globais para estatísticas
stats = {'acertos': 0, 'erros': 0, 'total': 0}

# Estado inicial do tabuleiro
def novo_tabuleiro():
    return ['b'] * 9

# Função para checar o estado real do jogo
def checar_estado_real(board):
    combinacoes = [
        [0,1,2], [3,4,5], [6,7,8],
        [0,3,6], [1,4,7], [2,5,8],
        [0,4,8], [2,4,6]
    ]
    for combo in combinacoes:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] != 'b':
            return f"{board[combo[0]]}_venceu"
    if 'b' not in board:
        return 'empate'
    return 'em_jogo'

# Função para traduzir estado
def traduzir_estado(estado):
    return {
        'x_venceu': 'Jogador X venceu',
        'o_venceu': 'Jogador O venceu',
        'empate': 'Empate',
        'em_jogo': 'Jogo em andamento'
    }.get(estado, estado)

# Função para IA jogar
import random
def jogada_ia(board):
    vazias = [i for i, v in enumerate(board) if v == 'b']
    if vazias:
        pos = random.choice(vazias)
        board[pos] = 'o'
    return board

# Função para prever estado com IA
def prever_estado(board):
    mapping = {'x': 1, 'o': 0, 'b': 2}
    feature_names = ['top-left', 'top-middle', 'top-right',
                    'middle-left', 'middle-middle', 'middle-right',
                    'bottom-left', 'bottom-middle', 'bottom-right']
    input_board = [mapping[v] for v in board]
    input_df = pd.DataFrame([input_board], columns=feature_names)
    encoded_pred = best_model.predict(input_df)[0]
    estado_predito = encoder.inverse_transform([encoded_pred])[0]
    return estado_predito

# Página principal
HTML = '''
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <title>Jogo da Velha com IA</title>
    <style>
        body { font-family: Arial; text-align: center; }
        table { margin: 20px auto; border-collapse: collapse; }
        td { width: 60px; height: 60px; font-size: 2em; text-align: center; border: 2px solid #333; cursor: pointer; }
        .status { margin: 20px; font-size: 1.2em; }
        .caixa { display: inline-block; padding: 10px 20px; border-radius: 8px; margin: 10px; font-size: 1.1em; }
        .ia { background: #e3f2fd; border: 2px solid #2196f3; }
        .real { background: #e8f5e9; border: 2px solid #43a047; }
        .acerto { color: #388e3c; font-weight: bold; }
        .erro { color: #d32f2f; font-weight: bold; }
        button { margin-top: 20px; padding: 10px 20px; font-size: 1em; }
    </style>
</head>
<body>
    <h1>Jogo da Velha com Classificação de IA</h1>
    <div class="status" id="status">{{ status|safe }}</div>
    <table>
        {% for i in range(3) %}
        <tr>
            {% for j in range(3) %}
            <td onclick="jogar({{ i*3 + j }})" id="c{{ i*3 + j }}">{{ board[i*3 + j] if board[i*3 + j] != 'b' else '' }}</td>
            {% endfor %}
        </tr>
        {% endfor %}
    </table>
    <button onclick="reiniciar()">Reiniciar</button>
    <script>
        let board = {{ board|tojson }};
        let fim = false;
        function jogar(pos) {
            if (fim) return;
            fetch('/jogar', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pos: pos, board: board })
            })
            .then(r => r.json())
            .then(data => {
                board = data.board;
                fim = data.fim;
                for (let i = 0; i < 9; i++) {
                    document.getElementById('c'+i).innerText = board[i] === 'b' ? '' : board[i];
                }
                document.getElementById('status').innerHTML = data.status;
            });
        }
        function reiniciar() {
            fetch('/reiniciar')
            .then(r => r.json())
            .then(data => {
                board = data.board;
                fim = false;
                for (let i = 0; i < 9; i++) {
                    document.getElementById('c'+i).innerText = '';
                }
                document.getElementById('status').innerHTML = data.status;
            });
        }
    </script>
</body>
</html>
'''

def montar_status(estado_predito):
    acuracia = stats['acertos'] / stats['total'] if stats['total'] > 0 else 0
    caixa_ia = f'<span class="caixa ia">Situação do jogo (IA): <b>{traduzir_estado(estado_predito)}</b></span>'
    estat = f"<br>Acertos: {stats['acertos']} | Erros: {stats['erros']} | Acurácia: {acuracia:.2%}"
    return f"{caixa_ia}{estat}"

@app.route("/")
def index():
    board = novo_tabuleiro()
    estado_predito = prever_estado(board)
    status = montar_status(estado_predito)
    return render_template_string(HTML, board=board, status=status)

@app.route("/jogar", methods=["POST"])
def jogar():
    data = request.get_json()
    board = data['board']
    pos = data['pos']
    estado_real = checar_estado_real(board)
    if estado_real in ['x_venceu', 'o_venceu', 'empate']:
        # Jogo já acabou, não faz mais nada
        estado_predito = prever_estado(board)
        status = montar_status(estado_predito)
        return jsonify({'board': board, 'status': status, 'fim': True})
    if board[pos] == 'b':
        board[pos] = 'x'
        if checar_estado_real(board) == 'em_jogo':
            board = jogada_ia(board)
    estado_predito = prever_estado(board)
    estado_real = checar_estado_real(board)
    if estado_predito == estado_real:
        stats['acertos'] += 1
    else:
        stats['erros'] += 1
    stats['total'] += 1
    status = montar_status(estado_predito)
    fim = estado_real in ['x_venceu', 'o_venceu', 'empate']
    return jsonify({'board': board, 'status': status, 'fim': fim})

@app.route("/reiniciar")
def reiniciar():
    board = novo_tabuleiro()
    estado_predito = prever_estado(board)
    status = montar_status(estado_predito)
    return jsonify({'board': board, 'status': status})

if __name__ == "__main__":
    app.run(debug=True) 