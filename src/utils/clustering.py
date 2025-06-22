import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

def elbow_method(X, max_k=10):
    """
    Implementa il metodo del gomito per trovare il k ottimale
    """
    inertias = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    return k_range, inertias

def plot_elbow(k_range, inertias, save_path=None):
    """
    Crea il grafico del metodo del gomito
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Numero di Cluster (k)', fontsize=12)
    plt.ylabel('Inerzia (WCSS)', fontsize=12)
    plt.title('Metodo del Gomito per la Determinazione del k Ottimale', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Evidenzia il punto k=4
    if 4 in k_range:
        idx = list(k_range).index(4)
        plt.plot(4, inertias[idx], 'ro', markersize=12, label='k=4 (ottimale)')
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grafico del gomito salvato in: {save_path}")
    
    plt.show()

def perform_clustering(X, k=4):
    """
    Esegue il clustering K-Means
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    return clusters, kmeans

def analyze_clusters(data, clusters, feature_names):
    """
    Analizza le caratteristiche di ogni cluster
    """
    # Aggiungi i cluster al dataset
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = clusters
    
    # Analisi per cluster
    cluster_analysis = {}
    
    for cluster_id in sorted(data_with_clusters['cluster'].unique()):
        cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
        
        analysis = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(data_with_clusters) * 100,
            'avg_rating': cluster_data['avg_rating'].mean() if 'avg_rating' in cluster_data.columns else None,
            'top_genres': {}
        }
        
        # Trova i generi più comuni nel cluster
        genre_cols = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 
                      'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                      'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
                      'Thriller', 'War', 'Western']
        
        for genre in genre_cols:
            if genre in cluster_data.columns:
                genre_percentage = cluster_data[genre].mean() * 100
                if genre_percentage > 10:  # Solo generi con almeno 10% di presenza
                    analysis['top_genres'][genre] = genre_percentage
        
        cluster_analysis[cluster_id] = analysis
    
    return cluster_analysis, data_with_clusters

def plot_cluster_distribution(cluster_analysis, save_path=None):
    """
    Crea un grafico a torta della distribuzione dei cluster
    """
    # Ordina gli item del dizionario in base alla chiave (ID del cluster)
    # Questo garantisce che dati e etichette siano sempre allineati
    sorted_items = sorted(cluster_analysis.items())
    
    # Crea le liste a partire dai dati ordinati
    sizes = [analysis['size'] for cluster_id, analysis in sorted_items]
    labels = [f'Cluster {cluster_id}' for cluster_id, analysis in sorted_items]
    
    colors = ['#66b3ff', '#ff9999', '#99ff99', '#ffcc99', '#c2c2f0'] 
    
    plt.figure(figsize=(10, 8))
    # Usiamo autopct per le percentuali e 'labels' per le etichette nella legenda
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
            startangle=90, textprops={'fontsize': 12})
    
    plt.title('Distribuzione dei Film nei Cluster Tematici', fontsize=16, pad=20)
    plt.legend(title="Cluster") # Aggiunge una legenda pulita
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grafico della distribuzione salvato in: {save_path}")
    
    plt.show()

def save_cluster_analysis(cluster_analysis, save_path):
    """
    Salva l'analisi dei cluster in un file di testo
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("ANALISI DEI CLUSTER\n")
        f.write("=" * 50 + "\n\n")
        
        for cluster_id, analysis in cluster_analysis.items():
            f.write(f"CLUSTER {cluster_id}\n")
            f.write("-" * 20 + "\n")
            f.write(f"Dimensione: {analysis['size']} film ({analysis['percentage']:.1f}%)\n")
            
            if analysis['avg_rating']:
                f.write(f"Rating medio: {analysis['avg_rating']:.3f}\n")
            
            f.write("Generi principali:\n")
            for genre, percentage in sorted(analysis['top_genres'].items(), 
                                          key=lambda x: x[1], reverse=True):
                f.write(f"  - {genre}: {percentage:.1f}%\n")
            
            # Interpretazione del cluster
            if cluster_id == 0:
                f.write("Interpretazione: Commedie Mainstream\n")
            elif cluster_id == 1:
                f.write("Interpretazione: Drammatici e Thriller\n")
            elif cluster_id == 2:
                f.write("Interpretazione: Avventura, Azione e Fantascienza\n")
            elif cluster_id == 3:
                f.write("Interpretazione: Film d'Animazione e per Bambini (Nicchia)\n")
            
            f.write("\n")
    
    print(f"Analisi dei cluster salvata in: {save_path}")

def main(data):
    """
    Funzione principale per il clustering
    """
    from data_preprocessing import prepare_clustering_data
    
    # Prepara i dati per il clustering
    X_clustering, feature_names = prepare_clustering_data(data)
    
    # Metodo del gomito
    print("Applicazione del metodo del gomito...")
    k_range, inertias = elbow_method(X_clustering, max_k=10)
    plot_elbow(k_range, inertias, 'results/clustering_elbow_plot.png')
    
    # Clustering con k=4
    print("Esecuzione del clustering con k=4...")
    clusters, kmeans_model, silhouette = perform_clustering(X_clustering, k=4)
    
    print(f"Silhouette Score: {silhouette:.3f}")
    
    # Analisi dei cluster
    print("Analisi dei cluster...")
    cluster_analysis, data_with_clusters = analyze_clusters(data, clusters, feature_names)
    
    # Visualizzazione
    plot_cluster_distribution(cluster_analysis, 'results/clustering_distribution_pie_chart.png')
    
    # Salva l'analisi
    save_cluster_analysis(cluster_analysis, 'results/cluster_analysis.txt')
    
    # Salva il dataset con i cluster
    output_path = 'data/movies_with_clusters.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data_with_clusters.to_csv(output_path, index=False)
    print(f"Dataset con cluster salvato in: {output_path}")
    
    return data_with_clusters, kmeans_model, cluster_analysis

if __name__ == "__main__":
    # Carica i dati preprocessati
    data = pd.read_csv('data/processed_movies.csv')
    main(data)