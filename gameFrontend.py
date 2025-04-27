import random
import pandas as pd

class GameFrontend:
    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder
        self.board = ['b'] * 9
        self.winning_combinations = [
            [0,1,2], [3,4,5], [6,7,8],  # linhas
            [0,3,6], [1,4,7], [2,5,8],  # colunas
            [0,4,8], [2,4,6]             # diagonais
        ]
        self.mapping = {'x': 1, 'o': 0, 'b': 2}
        self.feature_names = ['V1','V2','V3','V4','V5','V6','V7','V8','V9']
        self.stats = {'acertos': 0, 'erros': 0}

    def print_board(self):
        print("\n  0 | 1 | 2")
        print(" -----------")
        print("  3 | 4 | 5")
        print(" -----------")
        print("  6 | 7 | 8\n")
        
        print(f" {self.board[0]} | {self.board[1]} | {self.board[2]}")
        print("-----------")
        print(f" {self.board[3]} | {self.board[4]} | {self.board[5]}")
        print("-----------")
        print(f" {self.board[6]} | {self.board[7]} | {self.board[8]}\n")

    def human_move(self):
        while True:
            try:
                move = int(input("Sua jogada (0-8): "))
                if move < 0 or move > 8:
                    print("Posição inválida! Escolha de 0 a 8.")
                elif self.board[move] != 'b':
                    print("Posição ocupada! Tente novamente.")
                else:
                    self.board[move] = 'x'
                    return
            except ValueError:
                print("Entrada inválida! Digite um número.")

    def ai_move(self):
        empty_positions = [i for i, v in enumerate(self.board) if v == 'b']
        if empty_positions:
            move = random.choice(empty_positions)
            self.board[move] = 'o'
            print(f"IA jogou na posição {move}")

    def check_state(self):
        """Verifica o estado atual do jogo usando a IA"""
        # Prepara os dados para o modelo
        input_board = [self.mapping[v] for v in self.board]
        input_df = pd.DataFrame([input_board], columns=self.feature_names)
        
        # Faz a predição
        encoded_pred = self.model.predict(input_df)[0]
        estado_predito = self.encoder.inverse_transform([encoded_pred])[0]
        
        # Verifica o estado real
        estado_real = self.check_real_state()
        
        # Atualiza estatísticas
        if estado_predito == estado_real:
            self.stats['acertos'] += 1
        else:
            self.stats['erros'] += 1
        
        print(f"\nEstado predito: {self.translate_state(estado_predito)}")
        return estado_predito

    def check_real_state(self):
        """Verifica o estado real do jogo (sem usar o modelo)"""
        # Verifica vitórias
        for combo in self.winning_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != 'b':
                return f"{self.board[combo[0]]}_venceu"
        
        # Verifica empate
        if 'b' not in self.board:
            return 'empate'
        
        return 'em_jogo'

    def translate_state(self, estado):
        """Traduz os estados para português"""
        translations = {
            'x_venceu': 'Jogador X venceu',
            'o_venceu': 'Jogador O venceu',
            'empate': 'Empate',
            'em_jogo': 'Jogo em andamento'
        }
        return translations.get(estado, estado)

    def play(self):
        print("\n=== Jogo da Velha com Classificação de IA ===")
        print("Você é 'x', a IA é 'o'")
        
        while True:
            self.print_board()
            
            # Verifica estado antes da jogada humana
            estado = self.check_state()
            if estado != 'em_jogo':
                print(f"\nFim do jogo! {self.translate_state(estado)}")
                break
                
            # Jogada humana
            self.human_move()
            
            # Verifica estado após jogada humana
            estado_real = self.check_real_state()
            if estado_real != 'em_jogo':
                self.print_board()
                print(f"\nFim do jogo! {self.translate_state(estado_real)}")
                break
                
            # Jogada da IA
            self.ai_move()
            
            # Verifica estado após jogada da IA
            estado_real = self.check_real_state()
            if estado_real != 'em_jogo':
                self.print_board()
                print(f"\nFim do jogo! {self.translate_state(estado_real)}")
                break
        
        # Mostra estatísticas
        total = self.stats['acertos'] + self.stats['erros']
        acuracia = self.stats['acertos'] / total if total > 0 else 0
        print(f"\nDesempenho da IA:")
        print(f"Acertos: {self.stats['acertos']}")
        print(f"Erros: {self.stats['erros']}")
        print(f"Acurácia durante o jogo: {acuracia:.2%}")