# Jogo da Velha com Classificação de IA

Este projeto implementa um sistema de Inteligência Artificial para classificar o estado de um tabuleiro de Jogo da Velha (3x3) usando algoritmos de Machine Learning. O objetivo não é jogar contra a IA, mas sim fazer com que a IA classifique corretamente o estado do jogo em uma das quatro categorias:

- Jogo em andamento
- Jogador X venceu
- Jogador O venceu
- Empate

## Objetivo

A IA recebe como entrada o estado atual do tabuleiro e deve classificar corretamente o estado do jogo. O sistema foi desenvolvido para fins didáticos, seguindo as etapas de um trabalho prático de disciplina de IA.

## Tecnologias Utilizadas

- Python 3.8+
- Flask (frontend web)
- scikit-learn (modelos de ML)
- pandas, matplotlib (análise e visualização)
- ucimlrepo (para baixar o dataset base)

## Estrutura do Projeto

- `datasetLoader.py`: Carrega e prepara o dataset, gera exemplos sintéticos e faz o balanceamento das classes.
- `modelTrainer.py`: Treina e avalia múltiplos modelos de ML (k-NN, MLP, Decision Tree, Random Forest), gera tabela e gráfico comparativo.
- `webapp.py`: Frontend web para jogar e ver a classificação da IA em tempo real.
- `requirements.txt`: Dependências do projeto.

## Como rodar o projeto

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes do Python)
- Git

### No Linux

1. **Clone o repositório:**

   ```bash
   git clone <url-do-seu-repo>
   cd tic-tac-toe-ai
   ```

2. **Crie e ative um ambiente virtual:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instale as dependências:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Execute o sistema:**
   ```bash
   python webapp.py
   ```

### No Windows

1. **Clone o repositório:**

   ```cmd
   git clone <url-do-seu-repo>
   cd tic-tac-toe-ai
   ```

2. **Crie e ative um ambiente virtual:**

   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Instale as dependências:**

   ```cmd
   pip install -r requirements.txt
   ```

4. **Execute o sistema:**
   ```cmd
   python webapp.py
   ```

### Acessando a aplicação

Após executar o sistema, acesse no navegador:

- http://localhost:5000

### Solução de Problemas

#### Linux

- Se o comando python3 não for encontrado:
  ```bash
  sudo apt update
  sudo apt install python3
  ```
- Se o pip não estiver instalado:
  ```bash
  sudo apt install python3-pip
  ```

#### Windows

- Se o Python não for reconhecido, baixe e instale do [site oficial do Python](https://www.python.org/downloads/)
- Durante a instalação do Python no Windows, marque a opção "Add Python to PATH"
- Se precisar instalar o Git: [Download Git para Windows](https://git-scm.com/download/win)

## Funcionamento

- O sistema treina automaticamente 4 modelos de ML e seleciona o de melhor desempenho.
- O frontend web permite que o usuário jogue contra a IA (que joga aleatoriamente) e veja a classificação do estado do jogo feita pela IA.
- O sistema mostra estatísticas de acertos, erros e acurácia da IA durante a sessão.

## Avaliação dos Modelos

- São treinados e comparados: k-NN, MLP, Decision Tree e Random Forest.
- O sistema gera uma tabela e um gráfico de comparação de acurácia, precision, recall e F1-score.
- O melhor modelo é usado no frontend.

## Exemplo de uso

- Ao jogar, a IA classifica o estado do tabuleiro após cada jogada.
- Exemplo de saída:
  - Situação do jogo (IA): Jogador X venceu
  - Acertos: 15 | Erros: 2 | Acurácia: 88.24%

## Créditos

- Projeto desenvolvido para fins acadêmicos.
- Dataset base: [UCI Machine Learning Repository - Tic-Tac-Toe Endgame Data Set](https://archive.ics.uci.edu/dataset/101/tic+tac+toe+endgame)
