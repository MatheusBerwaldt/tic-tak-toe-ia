from datasetLoader import DatasetLoader
from modelTrainer import ModelTrainer
from gameFrontend import GameFrontend
from sklearn.metrics import accuracy_score

def main():
    print("=== T1 - Tic Tac Toe com ML ===")
    
    # 1. Load and prepare dataset
    print("\nCarregando e preparando dataset...")
    loader = DatasetLoader()
    X_train, X_val, X_test, y_train, y_val, y_test, encoder = loader.load_data()
    
    # 2. Train and evaluate models
    print("\nTreinando e avaliando modelos...")
    trainer = ModelTrainer()
    trainer.train_and_evaluate(X_train, X_val, y_train, y_val)
    trainer.print_results()
    
    # 3. Get best model
    best_model = trainer.get_best_model()
    print(f"\nMelhor modelo selecionado: {type(best_model).__name__}")
    
    # 4. Test with test set
    test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    print(f"Acurácia no conjunto de teste: {test_accuracy:.4f}")
    
    # 5. Play game
    print("\nIniciando jogo...")
    game = GameFrontend(best_model, encoder)
    game.play()
    
    # 6. Show performance metrics
    metrics = game.get_performance_metrics()
    print("\n=== Métricas de Desempenho da IA durante o jogo ===")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")

if __name__ == "__main__":
    # main()  # Comentado para não rodar o frontend de terminal
    pass