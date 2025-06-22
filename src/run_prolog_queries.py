import os
import pandas as pd
from utils.prolog_interface import create_prolog_knowledge_base, execute_prolog_query

def main():
    """
    Script per dimostrare le capacità di ragionamento avanzato
    della knowledge base Prolog arricchita.
    """
    csv_path = 'data/movies_with_clusters.csv'
    if not os.path.exists(csv_path):
        print(f"File '{csv_path}' non trovato. Esegui prima 'main.py' per generarlo.")
        return

    # Definiamo i generi per coerenza con il training
    genre_cols = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                  'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                  'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    # Crea la KB con fatti e regole avanzate
    prolog_kb = create_prolog_knowledge_base(csv_path, genre_cols)

    print("\n" + "="*80)
    print(" DIMOSTRAZIONE DEL RAGIONAMENTO PROLOG AVANZATO")
    print("="*80 + "\n")

    # --- Query 1: Pattern Matching Semplice ---
    print("--- 1. Query di base: Trova film nel Cluster 0 ---")
    query1 = "film_info(Titolo, Generi, 0)"
    results1 = execute_prolog_query(prolog_kb, query1)
    print("Primi 3 risultati:")
    for res in results1[:3]:
        print(f" - Film: {res['Titolo']}, Generi: {res['Generi']}")

    # --- Query 2: Raccomandazione Base ---
    film_scelto = 'pulp_fiction'
    print(f"\n--- 2. Raccomandazioni base per '{film_scelto}' ---")
    query2 = f"consiglia_film_base('{film_scelto}', FilmConsigliato)"
    results2 = execute_prolog_query(prolog_kb, query2)
    print("Prime 3 raccomandazioni base:")
    for res in results2[:3]:
        print(f" - {res['FilmConsigliato']}")

    # --- Query 3: Raccomandazione con Qualità ---
    print(f"\n--- 3. Raccomandazioni di qualità superiore per '{film_scelto}' ---")
    query3 = f"consiglia_film_qualita('{film_scelto}', FilmConsigliato, Motivo)"
    results3 = execute_prolog_query(prolog_kb, query3)
    print("Raccomandazioni di qualità:")
    for res in results3[:3]:
        print(f" - {res['FilmConsigliato']} (Motivo: {res['Motivo']})")

    # --- Query 4: Raccomandazioni Cross-Cluster ---
    print(f"\n--- 4. Raccomandazioni cross-cluster per '{film_scelto}' ---")
    query4 = f"consiglia_cross_cluster('{film_scelto}', FilmConsigliato, Motivo)"
    results4 = execute_prolog_query(prolog_kb, query4)
    print("Prime 3 raccomandazioni cross-cluster:")
    for res in results4[:3]:
        print(f" - {res['FilmConsigliato']} (Motivo: {res['Motivo']})")

    # --- Query 5: Classificazione Intelligente ---
    print(f"\n--- 5. Che tipo di film è '{film_scelto}'? ---")
    query5 = f"tipo_film('{film_scelto}', Tipo)"
    results5 = execute_prolog_query(prolog_kb, query5)
    if results5:
        for res in results5:
            print(f" - {film_scelto} è classificato come: {res['Tipo']}")
    else:
        print(f" - {film_scelto} non rientra in nessuna categoria predefinita")

    # --- Query 6: Trova Blockbuster ---
    print("\n--- 6. Trova alcuni blockbuster nel dataset ---")
    query6 = "tipo_film(Film, 'blockbuster')"
    results6 = execute_prolog_query(prolog_kb, query6)
    print("Primi 5 blockbuster identificati:")
    for res in results6[:5]:
        print(f" - {res['Film']}")

    # --- Query 7: Film d'Autore ---
    print("\n--- 7. Trova film d'autore ---")
    query7 = "tipo_film(Film, 'film_d_autore')"
    results7 = execute_prolog_query(prolog_kb, query7)
    print("Primi 5 film d'autore:")
    for res in results7[:5]:
        print(f" - {res['Film']}")

    # --- Query 8: Raccomandazioni per Era ---
    print(f"\n--- 8. Raccomandazioni per era per '{film_scelto}' ---")
    query8 = f"consiglia_per_era('{film_scelto}', FilmConsigliato, Motivo)"
    results8 = execute_prolog_query(prolog_kb, query8)
    print("Prime 3 raccomandazioni per era:")
    for res in results8[:3]:
        print(f" - {res['FilmConsigliato']} (Motivo: {res['Motivo']})")

    # --- Query 9: Meta-ragionamento: Raccomandazioni Forti ---
    print(f"\n--- 9. Raccomandazioni più forti per '{film_scelto}' (con punteggio) ---")
    query9 = f"raccomandazione_forte('{film_scelto}', FilmConsigliato, Punteggio)"
    results9 = execute_prolog_query(prolog_kb, query9)
    if results9:
        # Ordina per punteggio (simulato qui, in realtà dovresti usare Prolog per l'ordinamento)
        print("Top 3 raccomandazioni forti:")
        for res in results9[:3]:
            print(f" - {res['FilmConsigliato']} (Punteggio: {res['Punteggio']})")

    # --- Query 10: Analisi Macro-Generi ---
    print("\n--- 10. Analisi macro-generi ---")
    print("Film di azione nel dataset:")
    query10 = "macro_genere(Genere, azione), film_info(Film, Generi, _), member(Genere, Generi)"
    results10 = execute_prolog_query(prolog_kb, query10)
    unique_films = list(set([res['Film'] for res in results10]))
    print(f"Trovati {len(unique_films)} film di azione. Primi 5:")
    for film in unique_films[:5]:
        print(f" - {film}")

    print("\n" + "="*80)
    print(" DIMOSTRAZIONE COMPLETATA")
    print(" La KB Prolog ora supporta:")
    print(" - Ragionamento gerarchico (macro-generi)")
    print(" - Classificazione intelligente dei film")
    print(" - Raccomandazioni multi-criterio")
    print(" - Meta-ragionamento con punteggi")
    print(" - Ragionamento temporale (ere)")
    print("="*80)

if __name__ == "__main__":
    main()