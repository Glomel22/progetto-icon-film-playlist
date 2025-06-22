from pyswip import Prolog
import pandas as pd
import os
import re
from tqdm import tqdm

def _sanitize_prolog_atom(text: str) -> str:
    """
    Pulisce una stringa per renderla un atomo Prolog valido e sicuro.
    """
    if not isinstance(text, str) or not text.strip():
        return "'sconosciuto'"
    
    # Converte in minuscolo e rimuove apici che potrebbero rompere la sintassi
    text = text.lower().replace("'", "").replace('"', "")
    # Rimuove l'anno tra parentesi (es. 'toy story (1995)' -> 'toy story')
    text = re.sub(r'\s*\(\d{4}\)\s*$', '', text).strip()
    # Sostituisce caratteri non alfanumerici (eccetto spazi e underscore) con underscore
    text = re.sub(r'[^\w\s_]+', '_', text)
    # Sostituisce spazi con underscore e collassa underscore multipli
    text = re.sub(r'\s+', '_', text).replace('-', '_')
    text = re.sub(r'_+', '_', text).strip('_')

    if not text:
        return "'sconosciuto'"
    
    # Racchiude sempre tra apici per sicurezza
    return f"'{text}'"

def create_prolog_knowledge_base(csv_file_path: str, genre_columns: list) -> Prolog:
    """
    Crea una knowledge base Prolog a partire da un file CSV,
    arricchendola con regole di ragionamento complesse.
    """
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"File CSV non trovato: {csv_file_path}")

    prolog = Prolog()
    df = pd.read_csv(csv_file_path)
    existing_genres = [col for col in genre_columns if col in df.columns]

    print(f"Creazione della knowledge base Prolog da '{csv_file_path}'...")

    # --- FASE 1: AFFERMAZIONE DEI FATTI DI BASE ---
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Asserzione fatti Prolog"):
        prolog_title = _sanitize_prolog_atom(row['title'])
        cluster_val = int(row['cluster'])
        rating_discrete = int(row['rating_discreto']) if 'rating_discreto' in row else 1
        rating_norm = float(row['rating_norm']) if 'rating_norm' in row else 0.5
        year_cat = int(row['release_year_categoria']) if 'release_year_categoria' in row else 1
        
        active_genres = [_sanitize_prolog_atom(genre) for genre in existing_genres if row[genre] == 1]
        genres_list_str = f"[{', '.join(active_genres)}]"
        
        # Fatti base più ricchi
        prolog.assertz(f"film_info({prolog_title}, {genres_list_str}, {cluster_val})")
        prolog.assertz(f"film_rating({prolog_title}, {rating_discrete}, {rating_norm})")
        prolog.assertz(f"film_era({prolog_title}, {year_cat})")

    # --- FASE 2: DEFINIZIONE TASSONOMIE E GERARCHIE ---
    print("Definizione tassonomie di generi...")
    
    # Gerarchia di generi - Macro-categorie
    prolog.assertz("macro_genere('action', azione)")
    prolog.assertz("macro_genere('adventure', azione)")
    prolog.assertz("macro_genere('thriller', azione)")
    prolog.assertz("macro_genere('crime', azione)")
    
    prolog.assertz("macro_genere('comedy', intrattenimento)")
    prolog.assertz("macro_genere('animation', intrattenimento)")
    prolog.assertz("macro_genere('children_s', intrattenimento)")
    prolog.assertz("macro_genere('musical', intrattenimento)")
    
    prolog.assertz("macro_genere('drama', drammatico)")
    prolog.assertz("macro_genere('romance', drammatico)")
    prolog.assertz("macro_genere('war', drammatico)")
    
    prolog.assertz("macro_genere('horror', specialistico)")
    prolog.assertz("macro_genere('sci_fi', specialistico)")
    prolog.assertz("macro_genere('fantasy', specialistico)")
    prolog.assertz("macro_genere('film_noir', specialistico)")
    prolog.assertz("macro_genere('mystery', specialistico)")
    prolog.assertz("macro_genere('western', specialistico)")
    prolog.assertz("macro_genere('documentary', specialistico)")

    # --- FASE 3: REGOLE COMPLESSE DI RAGIONAMENTO ---
    print("Asserzione delle regole di ragionamento Prolog...")
    
    # Predicati helper per la manipolazione di liste
    prolog.assertz("member(X, [X|_])")
    prolog.assertz("member(X, [_|T]) :- member(X, T)")
    
    prolog.assertz("intersect([], _, [])")
    prolog.assertz("intersect([H|T], L2, [H|R]) :- member(H, L2), intersect(T, L2, R)")
    prolog.assertz("intersect([H|T], L2, R) :- \\+ member(H, L2), intersect(T, L2, R)")
    
    prolog.assertz("list_length([], 0)")
    prolog.assertz("list_length([_|T], N) :- list_length(T, N1), N is N1 + 1")
    
    # Predicato per calcolare affinità di genere
    prolog.assertz("""
        affinita_genere(Genere1, Genere2, alta) :-
            macro_genere(Genere1, Categoria),
            macro_genere(Genere2, Categoria),
            Genere1 \\= Genere2
    """)
    
    prolog.assertz("""
        affinita_genere(Genere, Genere, perfetta)
    """)

    # REGOLA 1: Raccomandazione base (quella che avevi già)
    prolog.assertz("""
        consiglia_film_base(FilmDiPartenza, FilmConsigliato) :-
            film_info(FilmDiPartenza, GeneriPartenza, Cluster),
            film_info(FilmConsigliato, GeneriConsigliato, Cluster),
            FilmDiPartenza \\= FilmConsigliato,
            intersect(GeneriPartenza, GeneriConsigliato, GeneriComuni),
            list_length(GeneriComuni, NumGeneriComuni),
            NumGeneriComuni >= 2
    """)

    # REGOLA 2: Raccomandazione avanzata con qualità
    prolog.assertz("""
        consiglia_film_qualita(FilmDiPartenza, FilmConsigliato, Motivo) :-
            film_info(FilmDiPartenza, GeneriPartenza, Cluster),
            film_info(FilmConsigliato, GeneriConsigliati, Cluster),
            film_rating(FilmDiPartenza, RatingPart, _),
            film_rating(FilmConsigliato, RatingCons, _),
            FilmDiPartenza \\= FilmConsigliato,
            intersect(GeneriPartenza, GeneriConsigliati, GeneriComuni),
            list_length(GeneriComuni, NumGeneriComuni),
            NumGeneriComuni >= 2,
            RatingCons >= RatingPart,
            Motivo = 'stesso_cluster_qualita_superiore'
    """)

    # REGOLA 3: Raccomandazione cross-cluster per macro-generi affini
    prolog.assertz("""
        consiglia_cross_cluster(FilmDiPartenza, FilmConsigliato, Motivo) :-
            film_info(FilmDiPartenza, GeneriPartenza, Cluster1),
            film_info(FilmConsigliato, GeneriConsigliati, Cluster2),
            film_rating(FilmConsigliato, Rating, _),
            Cluster1 \\= Cluster2,
            FilmDiPartenza \\= FilmConsigliato,
            Rating >= 2,
            member(GenereP, GeneriPartenza),
            member(GenereC, GeneriConsigliati),
            affinita_genere(GenereP, GenereC, _),
            Motivo = 'generi_affini_alta_qualita'
    """)

    # REGOLA 4: Raccomandazione per mood/era
    prolog.assertz("""
        consiglia_per_era(FilmDiPartenza, FilmConsigliato, Motivo) :-
            film_era(FilmDiPartenza, Era),
            film_era(FilmConsigliato, Era),
            film_info(FilmDiPartenza, GeneriPartenza, _),
            film_info(FilmConsigliato, GeneriConsigliati, _),
            film_rating(FilmConsigliato, Rating, _),
            FilmDiPartenza \\= FilmConsigliato,
            Rating >= 1,
            intersect(GeneriPartenza, GeneriConsigliati, GeneriComuni),
            list_length(GeneriComuni, NumGeneriComuni),
            NumGeneriComuni >= 1,
            Motivo = 'stessa_era_generi_compatibili'
    """)

    # REGOLA 5: Classificazione intelligente del tipo di film
    prolog.assertz("""
        tipo_film(Film, 'blockbuster') :-
            film_info(Film, Generi, _),
            film_rating(Film, Rating, _),
            (member('action', Generi); member('adventure', Generi); member('sci_fi', Generi)),
            Rating >= 1
    """)

    prolog.assertz("""
        tipo_film(Film, 'film_d_autore') :-
            film_info(Film, Generi, _),
            film_rating(Film, Rating, _),
            (member('drama', Generi); member('film_noir', Generi)),
            Rating >= 2
    """)

    prolog.assertz("""
        tipo_film(Film, 'intrattenimento_familiare') :-
            film_info(Film, Generi, _),
            (member('animation', Generi); member('children_s', Generi); member('comedy', Generi))
    """)

    prolog.assertz("""
        tipo_film(Film, 'genere_specialistico') :-
            film_info(Film, Generi, _),
            (member('horror', Generi); member('western', Generi); member('documentary', Generi))
    """)

    # REGOLA 6: Raccomandazione intelligente unificata
    prolog.assertz("""
        consiglia_film(FilmDiPartenza, FilmConsigliato) :-
            consiglia_film_base(FilmDiPartenza, FilmConsigliato)
    """)

    prolog.assertz("""
        consiglia_film(FilmDiPartenza, FilmConsigliato) :-
            consiglia_film_qualita(FilmDiPartenza, FilmConsigliato, _)
    """)

    prolog.assertz("""
        consiglia_film(FilmDiPartenza, FilmConsigliato) :-
            consiglia_cross_cluster(FilmDiPartenza, FilmConsigliato, _)
    """)

    # REGOLA 7: Meta-ragionamento sulla qualità delle raccomandazioni
    prolog.assertz("""
        raccomandazione_forte(FilmDiPartenza, FilmConsigliato, Punteggio) :-
            consiglia_film_base(FilmDiPartenza, FilmConsigliato),
            film_rating(FilmConsigliato, Rating, _),
            film_info(FilmDiPartenza, GeneriP, _),
            film_info(FilmConsigliato, GeneriC, _),
            intersect(GeneriP, GeneriC, Comuni),
            list_length(Comuni, NumComuni),
            Punteggio is Rating * NumComuni
    """)
            
    print("Knowledge base Prolog creata e arricchita con regole complesse.")
    return prolog

def execute_prolog_query(prolog_instance: Prolog, query_string: str) -> list:
    """ Esegue una query sulla knowledge base Prolog fornita. """
    print(f"\n>>> Esecuzione query Prolog: {query_string}")
    try:
        results = list(prolog_instance.query(query_string))
        if not results:
            print("Nessun risultato trovato per la query.")
        else:
            print(f"Trovati {len(results)} risultati.")
        return results
    except Exception as e:
        print(f"Errore durante l'esecuzione della query '{query_string}': {e}")
        return []