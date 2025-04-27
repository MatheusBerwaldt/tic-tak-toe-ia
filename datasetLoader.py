import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

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
        
        # Transforma para classificação multiclasse (4 estados)
        df['estado'] = df.apply(self.determinar_estado, axis=1)
        
        # Balanceamento (200 exemplos de cada classe)
        samples_per_class = 200
        balanced_df = pd.concat([
            df[df['estado'] == 'x_venceu'].sample(samples_per_class, random_state=42),
            df[df['estado'] == 'o_venceu'].sample(samples_per_class, random_state=42),
            df[df['estado'] == 'empate'].sample(samples_per_class, random_state=42),
            df[df['estado'] == 'em_jogo'].sample(samples_per_class, random_state=42)
        ])
        
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