import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import pickle
import os



def train_decision_tree(X_train, y_train):
    """Addestra un Decision Tree e restituisce l'oggetto GridSearchCV."""
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search # RESTITUISCE L'OGGETTO INTERO

def train_random_forest(X_train, y_train):
    """Addestra un Random Forest e restituisce l'oggetto GridSearchCV."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search # RESTITUISCE L'OGGETTO INTERO

def train_logistic_regression(X_train, y_train):
    """Addestra una Logistic Regression e restituisce l'oggetto GridSearchCV."""
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    lr = LogisticRegression(random_state=42, max_iter=1000)
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search # RESTITUISCE L'OGGETTO INTERO

def evaluate_model(model, X_test, y_test):
    """Valuta le performance di un modello."""
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0)
    }
    return metrics, y_pred

def plot_confusion_matrix(y_test, y_pred, model_name, save_path=None):
    """Crea la matrice di confusione."""
    labels = sorted(list(set(y_test) | set(y_pred)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Cluster {i}' for i in labels],
                yticklabels=[f'Cluster {i}' for i in labels])
    plt.title(f'Matrice di Confusione - {model_name}')
    plt.xlabel('Predetto')
    plt.ylabel('Reale')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matrice di confusione salvata in: {save_path}")
    plt.close()

def plot_feature_importance(model, feature_names, model_name, save_path=None):
    """Visualizza l'importanza delle feature (per modelli ad albero)."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.title(f'Importanza delle Feature - {model_name}')
        plt.xlabel('Feature')
        plt.ylabel('Importanza')
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grafico importanza feature salvato in: {save_path}")
        plt.close()

def save_results(results, save_path):
    """
    Salva i risultati della classificazione in modo robusto.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("RISULTATI DELLA CLASSIFICAZIONE\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name, result in results.items():
            f.write(f"--- {model_name.upper()} ---\n")
            f.write("-" * 30 + "\n")
            
            # Controlla se la chiave 'best_params' esiste prima di accedervi
            if 'best_params' in result:
                f.write(f"Parametri ottimali: {result['best_params']}\n")
            
            f.write(f"Accuratezza: {result['metrics']['accuracy']:.4f}\n")
            f.write(f"Precision (Macro): {result['metrics']['precision_macro']:.4f}\n")
            f.write(f"Recall (Macro): {result['metrics']['recall_macro']:.4f}\n")
            f.write(f"F1-Score (Macro): {result['metrics']['f1_macro']:.4f}\n\n")
            
            if 'classification_report' in result:
                f.write("Report di Classificazione per Classe:\n")
                f.write(result['classification_report'])
                f.write("\n")
            
            f.write("-" * 50 + "\n\n")

    print(f"Risultati salvati in: {save_path}")

def train_and_evaluate_models(X_train, y_train, X_test, y_test, feature_names, use_smote=True):
    """
    Funzione orchestratrice per la classificazione.
    Addestra, valuta, salva modelli e include un baseline.
    """
    if use_smote:
        print("\nApplicazione di SMOTE per bilanciare il training set...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"Distribuzione classi nel training set dopo SMOTE:\n{pd.Series(y_train).value_counts()}")

    results = {}
    models_to_train = {
        "DecisionTree": train_decision_tree,
        "RandomForest": train_random_forest,
        "LogisticRegression": train_logistic_regression
    }

    # Addestramento Baseline
    print("\nAddestramento Baseline (Dummy Classifier)...")
    dummy_clf = DummyClassifier(strategy='most_frequent')
    dummy_clf.fit(X_train, y_train)
    dummy_metrics, dummy_preds = evaluate_model(dummy_clf, X_test, y_test)
    results['Baseline'] = {
        'model': dummy_clf,
        'metrics': dummy_metrics,
        'predictions': dummy_preds,
        'classification_report': classification_report(y_test, dummy_preds, zero_division=0)
    }
    print(f"Baseline (Most Frequent) - F1-Score (Macro): {dummy_metrics['f1_macro']:.4f}")

    # Addestramento modelli reali
    for name, train_func in models_to_train.items():
        print(f"\nAddestramento e CV per {name}...")
        
        # 1. Chiamiamo train_func, che restituisce l'UNICO oggetto GridSearchCV
        grid_search = train_func(X_train, y_train)
        
        # 2. Estraiamo il modello migliore e i parametri DA quell'oggetto
        model = grid_search.best_estimator_
        params = grid_search.best_params_
        # ######################################
        
        # 3. Valutiamo il modello migliore sul test set
        metrics, predictions = evaluate_model(model, X_test, y_test)
        
        results[name] = {
            'model': model,
            'best_params': params,
            'metrics': metrics,
            'predictions': predictions,
            'classification_report': classification_report(y_test, predictions, zero_division=0)
        }
        print(f"{name} - F1-Score (Macro) su Test Set: {metrics['f1_macro']:.4f}")

    print("\nCreazione delle visualizzazioni...")
    for model_name, result in results.items():
        plot_confusion_matrix(y_test, result['predictions'], model_name,
                              f'results/confusion_matrix_{model_name.lower()}.png')
    
    if 'RandomForest' in results:
        plot_feature_importance(results['RandomForest']['model'], feature_names, 'Random Forest',
                              'results/feature_importance_rf.png')
    
    save_results(results, 'results/classification_results.txt')
    
    os.makedirs('models', exist_ok=True) # Salvo in /models per coerenza
    for model_name, result in results.items():
        if model_name != 'Baseline': # Non salviamo il modello dummy
            model_path = f'models/{model_name.lower()}_classifier_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f)
            print(f"Modello {model_name} salvato in: {model_path}")
    
    return results ### Restituiamo i risultati per l'analisi errori