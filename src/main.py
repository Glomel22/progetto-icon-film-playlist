import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Importa le funzioni helper dai moduli utils
from utils.data_preprocessing import load_and_preprocess_data
from utils.clustering import perform_clustering, analyze_clusters, plot_cluster_distribution, save_cluster_analysis, elbow_method, plot_elbow
from utils.classification import train_and_evaluate_models
# ### MODIFICATO ### Importiamo le funzioni aggiornate per la BN
from utils.bayesian_network import create_bayesian_network_model, evaluate_bayesian_network_robust, perform_bayesian_inference

def main():
    """
    Funzione principale che orchestra l'intera pipeline del progetto,
    dalla preparazione dei dati alla valutazione dei modelli avanzati.
    """
    
    # --- FASE 1: PREPROCESSING DEI DATI ---
    print("="*80)
    print(" 1. PREPROCESSING DEI DATI")
    print("="*80)
    processed_df = load_and_preprocess_data(dataset_path='data/ml-100k')
    print("Preprocessing completato.\n")

    # --- FASE 2: SPLIT TRAIN/TEST (PASSO CRUCIALE ANTI-LEAKAGE) ---
    print("="*80)
    print(" 2. DIVISIONE DEL DATASET IN TRAINING E TEST SET")
    print("="*80)
    train_df, test_df = train_test_split(processed_df, test_size=0.2, random_state=42, stratify=processed_df['rating_discreto'])
    print(f"Dataset diviso in: {len(train_df)} film (training) e {len(test_df)} film (test).\n")

    # Definiamo le colonne dei generi e le feature per ogni fase
    genre_cols = [
        'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
        'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    features_for_clustering = genre_cols + ['rating_norm']
    features_for_classification = genre_cols + ['rating_discreto']
    
    X_train_cluster_features = train_df[features_for_clustering]
    X_test_cluster_features = test_df[features_for_clustering]

    # --- FASE 3: CLUSTERING (SOLO SUL TRAINING SET) ---
    print("="*80)
    print(" 3. SCOPERTA DEI CLUSTER TRAMITE K-MEANS (SU TRAINING SET)")
    print("="*80)
    
    # K-Means è sensibile alla scala, quindi scaliamo i dati
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_cluster_features)

    # Applichiamo il Metodo del Gomito per determinare k ottimale
    print("Applicazione del Metodo del Gomito...")
    k_range, inertias = elbow_method(X_train_scaled, max_k=10)
    plot_elbow(k_range, inertias, 'results/clustering_elbow_plot.png')
    print("Grafico del gomito salvato in 'results/'.\n")
    
    # Eseguiamo il clustering con k=4 e otteniamo il modello addestrato
    train_clusters_labels, kmeans_model = perform_clustering(X_train_scaled, k=4)
    train_df['cluster'] = train_clusters_labels
    
    # Analizziamo e visualizziamo i cluster trovati
    print("Analisi dei cluster sul training set:")
    cluster_analysis, _ = analyze_clusters(train_df, train_clusters_labels, features_for_clustering)
    save_cluster_analysis(cluster_analysis, 'results/cluster_analysis.txt')
    plot_cluster_distribution(cluster_analysis, 'results/cluster_distribution_pie_chart.png')
    
    # Salviamo i modelli per poterli riutilizzare senza riaddestrare
    os.makedirs('models', exist_ok=True)
    pickle.dump(kmeans_model, open('models/kmeans_model.pkl', 'wb'))
    pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
    print("Clustering completato. Modelli K-Means e Scaler salvati.\n")

    # --- FASE 4: CLASSIFICAZIONE E VALIDAZIONE DEI CLUSTER ---
    print("="*80)
    print(" 4. ADDESTRAMENTO E VALUTAZIONE DEI CLASSIFICATORI")
    print("="*80)
    
    X_train_class_features = train_df[features_for_classification]
    y_train_class_labels = train_df['cluster']
    
    X_test_class_features = test_df[features_for_classification]
    
    # Usiamo il modello K-Means addestrato per PREDIRE i cluster per il test set
    X_test_scaled = scaler.transform(X_test_cluster_features)
    y_test_class_labels = kmeans_model.predict(X_test_scaled)
    test_df['cluster'] = y_test_class_labels # Aggiungiamo il cluster al test_df per analisi future
    
    # Addestriamo e valutiamo i modelli, catturando i risultati
    classification_results = train_and_evaluate_models(
        X_train=X_train_class_features,
        y_train=y_train_class_labels,
        X_test=X_test_class_features,
        y_test=y_test_class_labels,
        feature_names=features_for_classification
    )
    print("Classificazione completata. I risultati eccellenti validano la qualità dei cluster.\n")

    # --- FASE 5: RETE BAYESIANA (CON VALIDAZIONE ROBUSTA) ---
    print("="*80)
    print(" 5. MODELLAZIONE PROBABILISTICA CON RETE BAYESIANA")
    print("="*80)
    
    # 5.1 Addestramento del modello con Cross-Validation
    print("\n--- 5.1 Addestramento e Cross-Validation della Rete Bayesiana (su Training Set) ---")
    # ### MODIFICATO ### Chiamata alla funzione aggiornata
    bayesian_model, cv_results = create_bayesian_network_model(train_df, genre_cols=genre_cols)
    print("Modello finale addestrato su tutto il training set.")
    
    # 5.2 Valutazione quantitativa ROBUSTA del modello sul test set
    print("\n--- 5.2 Valutazione predittiva robusta della Rete Bayesiana su Test Set ---")
    bn_features = ['rating_discreto', 'release_year_categoria', 'cluster'] + genre_cols
    bn_features = [col for col in bn_features if col in train_df.columns]
    # ### MODIFICATO ### Chiamata alla funzione di valutazione aggiornata
    evaluate_bayesian_network_robust(bayesian_model, test_df, bn_features, cv_results)
    
    # 5.3 Esempi di inferenza (ragionamento probabilistico)
    print("\n--- 5.3 Esecuzione di inferenze di esempio ---")
    evidence1 = {'Comedy': 1, 'rating_discreto': 0} # Film comico con basso rating
    perform_bayesian_inference(bayesian_model, evidence1, 'cluster')
    
    evidence2 = {'Action': 1, 'Sci-Fi': 1, 'rating_discreto': 2} # Film Sci-Fi/Azione con alto rating
    perform_bayesian_inference(bayesian_model, evidence2, 'cluster')

    # --- FASE 6: PREPARAZIONE DATI FINALI PER PROLOG ---
    print("\n" + "="*80)
    print(" 6. PREPARAZIONE DATI FINALI PER LA KNOWLEDGE BASE PROLOG")
    print("="*80)
    
    # Applichiamo il modello K-Means finale all'intero dataset per avere etichette per tutti i film
    full_dataset_scaled = scaler.transform(processed_df[features_for_clustering])
    processed_df['cluster'] = kmeans_model.predict(full_dataset_scaled)
    # Salviamo il CSV che sarà letto dallo script Prolog
    processed_df.to_csv('data/movies_with_clusters.csv', index=False)
    print("Dataset completo con etichette cluster salvato in 'data/movies_with_clusters.csv'.")
    print("Questo file verrà utilizzato da 'run_prolog_queries.py'.")
    
    print("\n" + "="*80)
    print(" FLUSSO DI LAVORO DEL PROGETTO COMPLETATO CON SUCCESSO")
    print("="*80)

if __name__ == "__main__":
    main()