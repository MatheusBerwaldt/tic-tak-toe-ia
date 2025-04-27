from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

class ModelTrainer:
    def __init__(self):
        self.models = {
            "k-NN": KNeighborsClassifier(n_neighbors=5),
            "MLP": MLPClassifier(hidden_layer_sizes=(100,50), max_iter=2000),
            "Decision Tree": DecisionTreeClassifier(max_depth=10),
            "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10)
        }
        self.results = {}
        self.best_model = None

    def train_and_evaluate(self, X_train, X_val, y_train, y_val):
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # Armazena resultados detalhados
            self.results[name] = {
                'model': model,
                'accuracy': accuracy_score(y_val, y_pred),
                'report': classification_report(y_val, y_pred, output_dict=True)
            }
        
        # Seleciona o melhor modelo baseado na acurácia
        self.best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])

    def print_results(self):
        print("\n=== Resultados dos Modelos ===")
        for name, metrics in self.results.items():
            print(f"\n{name}:")
            print(f"Acurácia: {metrics['accuracy']:.4f}")
            print("Relatório:")
            print(pd.DataFrame(metrics['report']).transpose())
        
        print(f"\nMelhor modelo: {self.best_model[0]} (Acurácia: {self.best_model[1]['accuracy']:.4f})")

    def get_best_model(self):
        return self.best_model[1]['model']