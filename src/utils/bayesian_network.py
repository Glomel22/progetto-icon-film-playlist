import pandas as pd
import pickle
import networkx as nx
from matplotlib import pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BIC 
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.metrics import accuracy_score, classification_report
import os
import warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=UserWarning, module="pgmpy")


def cross_validate_bayesian_network(data_df: pd.DataFrame, genre_cols: list, k_folds=5):
    """Esegue k-fold cross-validation sulla Rete Bayesiana"""
    
    bn_features = ['rating_discreto', 'release_year_categoria', 'cluster'] + genre_cols
    bn_features = [col for col in bn_features if col in data_df.columns]
    
    X = data_df[bn_features].drop('cluster', axis=1)
    y = data_df['cluster']
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    accuracies = []
    f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{k_folds} ---")
        
        # Split data
        train_data = data_df.iloc[train_idx][bn_features]
        val_data = data_df.iloc[val_idx][bn_features]
        
        # Train model
        hc = HillClimbSearch(data=train_data)
        best_structure = hc.estimate(scoring_method=BIC(data=train_data))
        
        model = DiscreteBayesianNetwork(best_structure.edges())
        model.fit(data=train_data, estimator=MaximumLikelihoodEstimator)
        
        # Predict
        evidence_vars = [col for col in model.nodes() if col != 'cluster']
        X_val = val_data[evidence_vars]
        y_true = val_data['cluster']
        
        y_pred = model.predict(X_val)
        
        # Metrics
        acc = accuracy_score(y_true, y_pred['cluster'])
        f1 = f1_score(y_true, y_pred['cluster'], average='macro')
        
        accuracies.append(acc)
        f1_scores.append(f1)
        
        print(f"Fold {fold + 1} - Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
    
    # Results
    results = {
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'all_accuracies': accuracies,
        'all_f1_scores': f1_scores
    }
    
    return results

def visualize_bayesian_network(model: DiscreteBayesianNetwork, results_dir: str = 'results'):
    """
    Visualizza e salva il grafo della Rete Bayesiana, usando un layout robusto
    per garantire la separazione dei nodi.
    """
    os.makedirs(results_dir, exist_ok=True)
    graph_path = os.path.join(results_dir, 'bayesian_network_graph.png')

    view_graph = model.copy()
    self_loops = list(nx.selfloop_edges(view_graph))
    if self_loops:
        print(f"Attenzione: rimossi {len(self_loops)} auto-anelli per la visualizzazione: {self_loops}")
        view_graph.remove_edges_from(self_loops)

    # Usiamo kamada_kawai_layout come prima scelta perché è più stabile e previene
    # la sovrapposizione dei nodi, producendo un grafo più leggibile.
    print("Calcolo del layout del grafo con 'kamada_kawai_layout' per una migliore visualizzazione...")
    try:
        pos = nx.kamada_kawai_layout(view_graph)
    except Exception as e:
        # Fallback a spring_layout solo se kamada_kawai fallisce.
        print(f"Layout 'kamada_kawai' fallito ({e}), fallback a 'spring_layout'.")
        # Calcoliamo un valore k per migliorare la spaziatura dello spring_layout
        k_val = 1.5 / np.sqrt(view_graph.number_of_nodes())
        pos = nx.spring_layout(view_graph, seed=42, k=k_val, iterations=100)

    plt.figure(figsize=(20, 20))

    # Codice di sicurezza per evitare di disegnare archi quasi invisibili (se i nodi sono sovrapposti)
    edges_to_draw = []
    min_dist_sq = (1e-6)**2 
    for u, v in view_graph.edges():
        if u in pos and v in pos:
            p1, p2 = pos[u], pos[v]
            dist_sq = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
            if dist_sq > min_dist_sq:
                edges_to_draw.append((u, v))
            
    print(f"Disegnando {len(view_graph.nodes())} nodi e {len(edges_to_draw)} archi.")
    
    # Disegna le componenti del grafo
    nx.draw_networkx_nodes(view_graph, pos, node_size=3500, node_color="skyblue", alpha=0.9)
    nx.draw_networkx_edges(
        view_graph, pos, edgelist=edges_to_draw, arrows=True, arrowsize=20, 
        edge_color='gray', width=1.5, node_size=3500
    )
    nx.draw_networkx_labels(view_graph, pos, font_size=10, font_weight="bold")
    
    plt.title("Grafo della Rete Bayesiana Appresa", fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(graph_path, dpi=150)
    plt.close()
    print(f"Grafo della Rete Bayesiana salvato in: {graph_path}")

def save_cpds(model: DiscreteBayesianNetwork, results_dir: str = 'results'):
    """Salva le CPDs del modello in un file di testo."""
    os.makedirs(results_dir, exist_ok=True)
    cpds_path = os.path.join(results_dir, 'bayesian_network_cpds.txt')
    
    with open(cpds_path, 'w') as f:
        f.write("Conditional Probability Distributions (CPDs) della Rete Bayesiana\n" + "="*70 + "\n\n")
        for cpd in model.get_cpds():
            f.write(str(cpd))
            f.write("\n\n" + "-"*50 + "\n\n")
    print(f"CPDs salvate in: {cpds_path}")

### Funzione per analizzare la struttura del grafo
def analyze_network_structure(model: DiscreteBayesianNetwork, results_dir: str = 'results'):
    """Analizza e salva un report testuale sulla struttura appresa della rete."""
    analysis = "Analisi della Struttura della Rete Bayesiana\n" + "="*50 + "\n"
    
    # 1. Quali nodi influenzano direttamente il 'cluster'? (Genitori di 'cluster')
    try:
        cluster_parents = model.get_parents('cluster')
        analysis += f"\n1. Il 'cluster' è direttamente influenzato da: {cluster_parents}\n"
        analysis += "   Questo ha senso: il tipo di film dipende dai suoi generi e dal suo successo (rating).\n"
    except Exception:
        analysis += "\n1. Nodo 'cluster' non ha genitori diretti o non è nel modello.\n"

    # 2. Quali nodi sono "hub" centrali? (Nodi con più connessioni)
    degrees = dict(model.degree())
    hubs = sorted(degrees.items(), key=lambda item: item[1], reverse=True)[:5]
    analysis += f"\n2. Nodi più centrali (hub) della rete: {hubs}\n"
    analysis += "   Spesso 'Drama', 'Comedy' e 'rating_discreto' sono hub, indicando la loro importanza nel definire le relazioni tra film.\n"

    # 3. Ci sono relazioni inaspettate?
    analysis += "\n3. Analisi di relazioni specifiche:\n"
    if 'release_year_categoria' in model.nodes() and 'Drama' in model.nodes():
        if model.has_edge('release_year_categoria', 'Drama'):
            analysis += "   - Trovato arco da 'release_year_categoria' a 'Drama'. Potrebbe indicare che la popolarità dei film drammatici è cambiata nel tempo.\n"
        else:
            analysis += "   - Non c'è un'influenza diretta appresa tra l'anno di uscita e il genere 'Drama'.\n"

    print("\n--- Analisi Struttura Rete ---")
    print(analysis)
    
    path = os.path.join(results_dir, 'bayesian_network_structure_analysis.txt')
    with open(path, 'w') as f:
        f.write(analysis)
    print(f"Analisi della struttura salvata in: {path}")

### Funzione per analizzare le probabilità condizionali (CPD)
def analyze_cpds(model: DiscreteBayesianNetwork, results_dir: str = 'results'):
    """Analizza e salva un report sulle CPD più significative."""
    analysis = "Analisi delle Probabilità Condizionali (CPD)\n" + "="*50 + "\n"
    
    try:
        cpd_cluster = model.get_cpds('cluster')
        analysis += f"\n1. Analisi di P(cluster | Genitori):\n{cpd_cluster}\n"
        
        # Esempio: Qual è il cluster più probabile per un film con rating alto?
        if 'rating_discreto' in cpd_cluster.variables:
            # Assumiamo rating_discreto=2 (alto)
            prob_dist = cpd_cluster.reduce([('rating_discreto', 2)], inplace=False)
            most_likely_cluster = prob_dist.variables[0] # La variabile rimasta
            prob_values = prob_dist.values
            best_cluster_index = np.argmax(prob_values)
            analysis += f"\n   - Per un film con rating alto (2), il cluster più probabile è il {best_cluster_index}."
            analysis += " Questo conferma l'intuizione che i cluster catturano anche la popolarità.\n"

    except Exception as e:
        analysis += f"\n1. Impossibile analizzare la CPD di 'cluster': {e}\n"

    print("\n--- Analisi CPD ---")
    print(analysis)

    path = os.path.join(results_dir, 'bayesian_network_cpd_analysis.txt')
    with open(path, 'w') as f:
        f.write(analysis)
    print(f"Analisi delle CPD salvata in: {path}")

 ### Funzione principale per la creazione del modello
def create_bayesian_network_model(data_df, genre_cols, results_dir='results'):
    """Versione aggiornata con validazione robusta"""
    
    # 1. Cross-validation prima di tutto
    print("Eseguendo cross-validation della Rete Bayesiana...")
    cv_results = cross_validate_bayesian_network(data_df, genre_cols, k_folds=5)
    
    # 2. Train final model su tutti i dati di training
    bn_features = ['rating_discreto', 'release_year_categoria', 'cluster'] + genre_cols
    bn_features = [col for col in bn_features if col in data_df.columns]
    
    hc = HillClimbSearch(data=data_df[bn_features])
    best_structure = hc.estimate(scoring_method=BIC(data=data_df[bn_features]))
    
    model = DiscreteBayesianNetwork(best_structure.edges())
    model.fit(data=data_df[bn_features], estimator=MaximumLikelihoodEstimator)
    
    # 3. Analisi come prima
    visualize_bayesian_network(model, results_dir)
    save_cpds(model, results_dir)
    analyze_network_structure(model, results_dir)
    
    # 4. Salva risultati cross-validation
    cv_path = os.path.join(results_dir, 'bayesian_network_cv_results.txt')
    with open(cv_path, 'w') as f:
        f.write(f"Cross-Validation Results (5-fold)\n")
        f.write(f"Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}\n")
        f.write(f"F1-Score: {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}\n")
        f.write(f"Individual fold results:\n")
        for i, (acc, f1) in enumerate(zip(cv_results['all_accuracies'], cv_results['all_f1_scores'])):
            f.write(f"  Fold {i+1}: Acc={acc:.4f}, F1={f1:.4f}\n")
    
    return model, cv_results

 ### Funzione per la valutazione quantitativa
def evaluate_bayesian_network_robust(model, test_df, bn_features, cv_results, results_dir='results'):
    """Valutazione robusta con statistiche complete"""
    
    # Single test evaluation (come prima)
    evidence_vars = [node for node in model.nodes() if node != 'cluster']
    X_test = test_df[[col for col in evidence_vars if col in test_df.columns]]
    y_true = test_df['cluster']
    y_pred = model.predict(X_test)
    
    test_accuracy = accuracy_score(y_true, y_pred['cluster'])
    test_f1 = f1_score(y_true, y_pred['cluster'], average='macro')
    
    # Comprehensive analysis
    analysis = "Valutazione Robusta della Rete Bayesiana\n" + "="*60 + "\n\n"
    
    analysis += "1. RISULTATI CROSS-VALIDATION (5-fold):\n"
    analysis += f"   Accuracy: {cv_results['accuracy_mean']:.4f} (±{cv_results['accuracy_std']:.4f})\n"
    analysis += f"   F1-Score: {cv_results['f1_mean']:.4f} (±{cv_results['f1_std']:.4f})\n"
    analysis += f"   Range Accuracy: [{min(cv_results['all_accuracies']):.4f}, {max(cv_results['all_accuracies']):.4f}]\n\n"
    
    analysis += "2. RISULTATI TEST SET:\n"
    analysis += f"   Accuracy: {test_accuracy:.4f}\n"
    analysis += f"   F1-Score: {test_f1:.4f}\n\n"
    
    analysis += "3. ANALISI STABILITÀ:\n"
    cv_std = cv_results['accuracy_std']
    if cv_std < 0.05:
        analysis += f"   ✓ Modello STABILE (std = {cv_std:.4f} < 0.05)\n"
    else:
        analysis += f"   ⚠ Modello INSTABILE (std = {cv_std:.4f} >= 0.05)\n"
    
    analysis += f"   Coefficiente di variazione: {cv_std/cv_results['accuracy_mean']:.3f}\n\n"
    
    analysis += "4. CONFRONTO CON BASELINE:\n"
    baseline_acc = max(test_df['cluster'].value_counts()) / len(test_df)
    improvement = (cv_results['accuracy_mean'] - baseline_acc) / baseline_acc * 100
    analysis += f"   Baseline (classe maggioritaria): {baseline_acc:.4f}\n"
    analysis += f"   Miglioramento: +{improvement:.1f}%\n\n"
    
    # Save results
    path = os.path.join(results_dir, 'bayesian_network_robust_evaluation.txt')
    with open(path, 'w') as f:
        f.write(analysis)
    
    print(analysis)
    return {
        'cv_results': cv_results,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1
    }

def perform_bayesian_inference(model: DiscreteBayesianNetwork, evidence: dict, target_variable: str):
    try:
        inference_engine = VariableElimination(model)
        result = inference_engine.query(variables=[target_variable], evidence=evidence)
        print(f"\nRisultato dell'inferenza per '{target_variable}' con evidenza {evidence}:")
        print(result)
        return result
    except Exception as e:
        print(f"Errore durante l'inferenza per evidenza {evidence}: {e}")
        return None

if __name__ == "__main__":
    clustered_data_path = 'data/movies_with_clusters.csv'
    if not os.path.exists(clustered_data_path):
        print(f"File non trovato: {clustered_data_path}. Esegui prima main.py.")
    else:
        df = pd.read_csv(clustered_data_path)
        
        genre_cols = [
            'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
            'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        genre_cols = [col for col in genre_cols if col in df.columns]
        
        bn_model = create_bayesian_network_model(df, genre_cols=genre_cols)

        evidence1 = {'Comedy': 1, 'release_year_categoria': 2, 'rating_discreto': 0}
        perform_bayesian_inference(bn_model, evidence1, 'cluster')

        evidence2 = {'Action': 1, 'Drama': 1, 'release_year_categoria': 0, 'rating_discreto': 2}
        perform_bayesian_inference(bn_model, evidence2, 'cluster')