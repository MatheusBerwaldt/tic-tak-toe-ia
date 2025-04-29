import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo
import random

class DatasetLoader:
    def __init__(self):
        self.encoder = LabelEncoder()
    
    def load_data(self):
        # Carrega dataset original
        tic_tac_toe = fetch_ucirepo(id=101)
        df = pd.concat([tic_tac_toe.data.features, tic_tac_toe.data.targets], axis=1)
        
        # Renomeia colunas para nomes consistentes
        df.columns = ['top-left', 'top-middle', 'top-right',
                     'middle-left', 'middle-middle', 'middle-right',
                     'bottom-left', 'bottom-middle', 'bottom-right',
                     'resultado']
        
        # Geração de exemplos sintéticos de tabuleiros 'em_jogo' (válidos)
        def gerar_tabuleiro_em_jogo():
            while True:
                n_jogadas = random.randint(3, 8)
                board = ['b'] * 9
                jogadas = []
                for i in range(n_jogadas):
                    vazias = [idx for idx, v in enumerate(board) if v == 'b']
                    if not vazias:
                        break
                    pos = random.choice(vazias)
                    board[pos] = 'x' if i % 2 == 0 else 'o'
                # Checa se NÃO há vitória nem empate
                temp = pd.DataFrame([board], columns=[
                    'top-left', 'top-middle', 'top-right',
                    'middle-left', 'middle-middle', 'middle-right',
                    'bottom-left', 'bottom-middle', 'bottom-right'])
                estado = self.determinar_estado(temp.iloc[0])
                if estado == 'em_jogo':
                    temp['resultado'] = 'in_progress'
                    temp['estado'] = 'em_jogo'
                    return temp
        
        # Geração de exemplos sintéticos para todas as classes
        def gerar_vitoria(jogador):
            tabuleiros = []
            combinacoes = [
                [0,1,2], [3,4,5], [6,7,8],
                [0,3,6], [1,4,7], [2,5,8],
                [0,4,8], [2,4,6]
            ]
            for combo in combinacoes:
                for _ in range(70):
                    board = ['b'] * 9
                    for idx in combo:
                        board[idx] = jogador
                    # Preenche o resto aleatoriamente sem formar outra vitória
                    outros = [i for i in range(9) if i not in combo]
                    jogadas = ['x','o']*4
                    random.shuffle(jogadas)
                    for i, idx in enumerate(outros):
                        if board[idx] == 'b':
                            board[idx] = jogadas[i]
                            # Garante que não cria outra vitória
                            temp = pd.DataFrame([board], columns=[
                                'top-left', 'top-middle', 'top-right',
                                'middle-left', 'middle-middle', 'middle-right',
                                'bottom-left', 'bottom-middle', 'bottom-right'])
                            estado = self.determinar_estado(temp.iloc[0])
                            if estado != f'{jogador}_venceu':
                                board[idx] = 'b'
                    temp = pd.DataFrame([board], columns=[
                        'top-left', 'top-middle', 'top-right',
                        'middle-left', 'middle-middle', 'middle-right',
                        'bottom-left', 'bottom-middle', 'bottom-right'])
                    temp['resultado'] = 'synthetic'
                    temp['estado'] = f'{jogador}_venceu'
                    tabuleiros.append(temp)
            return pd.concat(tabuleiros, ignore_index=True)
        def gerar_empate():
            tabuleiros = []
            for _ in range(500):
                board = ['x']*5 + ['o']*4
                random.shuffle(board)
                temp = pd.DataFrame([board], columns=[
                    'top-left', 'top-middle', 'top-right',
                    'middle-left', 'middle-middle', 'middle-right',
                    'bottom-left', 'bottom-middle', 'bottom-right'])
                estado = self.determinar_estado(temp.iloc[0])
                if estado == 'empate':
                    temp['resultado'] = 'synthetic'
                    temp['estado'] = 'empate'
                    tabuleiros.append(temp)
            return pd.concat(tabuleiros, ignore_index=True)
        exemplos_x = gerar_vitoria('x')
        exemplos_o = gerar_vitoria('o')
        exemplos_empate = gerar_empate()
        exemplos_em_jogo = pd.concat([gerar_tabuleiro_em_jogo() for _ in range(500)], ignore_index=True)
        df = pd.concat([df, exemplos_x, exemplos_o, exemplos_empate, exemplos_em_jogo], ignore_index=True)
        
        # Transforma para classificação multiclasse (4 estados)
        df['estado'] = df.apply(self.determinar_estado, axis=1)
        
        # Balanceamento rigoroso: exatamente samples_per_class exemplos por classe
        samples_per_class = 500
        balanced_dfs = []
        for estado in ['x_venceu', 'o_venceu', 'empate', 'em_jogo']:
            df_estado = df[df['estado'] == estado]
            if len(df_estado) > 0:
                balanced_dfs.append(df_estado.sample(samples_per_class, random_state=42, replace=True))
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        # Codifica features (x->1, o->0, b->2)
        for col in balanced_df.columns[:-2]:  # Todas exceto resultado e estado
            balanced_df[col] = balanced_df[col].map({'x': 1, 'o': 0, 'b': 2})
        
        # Codifica target
        balanced_df['estado_encoded'] = self.encoder.fit_transform(balanced_df['estado'])
        
        # Separa features e target
        X = balanced_df.drop(['resultado', 'estado', 'estado_encoded'], axis=1)
        y = balanced_df['estado_encoded']
        
        # Divisão treino/val/teste
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        return X_train, X_val, X_test, y_train, y_val, y_test, self.encoder

    def determinar_estado(self, row):
        """Classifica cada linha em um dos 4 estados possíveis"""
        tabuleiro = row[['top-left', 'top-middle', 'top-right',
                        'middle-left', 'middle-middle', 'middle-right',
                        'bottom-left', 'bottom-middle', 'bottom-right']]
        
        # Verifica vitórias
        combinacoes = [
            ['top-left', 'top-middle', 'top-right'],
            ['middle-left', 'middle-middle', 'middle-right'],
            ['bottom-left', 'bottom-middle', 'bottom-right'],
            ['top-left', 'middle-left', 'bottom-left'],
            ['top-middle', 'middle-middle', 'bottom-middle'],
            ['top-right', 'middle-right', 'bottom-right'],
            ['top-left', 'middle-middle', 'bottom-right'],
            ['top-right', 'middle-middle', 'bottom-left']
        ]
        
        for combo in combinacoes:
            if (tabuleiro[combo[0]] == tabuleiro[combo[1]] == tabuleiro[combo[2]] and 
                tabuleiro[combo[0]] in ['x', 'o']):
                return f"{tabuleiro[combo[0]]}_venceu"
        
        # Verifica empate
        if all(tabuleiro[col] in ['x', 'o'] for col in tabuleiro.index):
            return 'empate'
        
        return 'em_jogo'