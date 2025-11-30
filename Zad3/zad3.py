import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os # Potrzebne do łączenia ścieżek plików

# --- IMPORTY BIBLIOTEK DO DANYCH ---
import kagglehub
from ucimlrepo import fetch_ucirepo

# Konfiguracja wykresów
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

def load_data_kaggle():  # sourcery skip: do-not-use-bare-except, use-contextlib-suppress
    print("--- Pobieranie danych z Kaggle (Mall Customers) ---")
    try:
        # 1. Pobieramy samą ścieżkę do folderu z danymi (zamiast load_dataset)
        path = kagglehub.dataset_download("shwetabh123/mall-customers")
        print("Ścieżka do pobranych plików:", path)

        # 2. Ręcznie wskazujemy plik CSV (to eliminuje błąd "Unsupported extension")
        csv_path = os.path.join(path, "Mall_Customers.csv")

        # 3. Ładujemy przez Pandas
        df = pd.read_csv(csv_path)
        print("Pobrano i załadowano Mall Customers.")

        # Wybieramy zmienne do klastrowania
        return df[['Annual Income (k$)', 'Spending Score (1-100)']], "Mall Customers"

    except Exception as e:
        print(f"Błąd przy pobieraniu z Kaggle: {e}")
        # Spróbujmy wylistować pliki w katalogu, jeśli coś pójdzie nie tak
        try:
            print("Pliki w katalogu:", os.listdir(path))
        except Exception:
            pass
        return None, None

def load_data_uci():
    print("\n--- Pobieranie danych z UCI (Wholesale Customers) ---")
    try:
        # fetch dataset 
        wholesale_customers = fetch_ucirepo(id=292) 
          
        # data (as pandas dataframes) 
        X = wholesale_customers.data.features.copy()
        
        print("Pobrano Wholesale Customers.")
        
        # Bezpieczne usuwanie (dla pewności, że nie ma zmiennych kategorycznych)
        X = X.drop(columns=['Channel', 'Region'], errors='ignore')
            
        return X, "Wholesale Customers"
    except Exception as e:
        print(f"Błąd przy pobieraniu z UCI: {e}")
        return None, None

def process_and_visualize(X, dataset_name):
    if X is None:
        return

    print(f"\nPrzetwarzanie: {dataset_name}")

    # 1. Standaryzacja danych
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Algorytm K-Means
    n_clusters = 5 if "Mall" in dataset_name else 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # 3. Algorytm DBSCAN
    # Dostrajanie parametrów dla lepszych wyników
    eps = 0.5 if "Mall" in dataset_name else 1.5
    min_samples = 5
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    # Statystyki
    print(f"  K-Means: Utworzono {n_clusters} grup.")
    n_noise = list(dbscan_labels).count(-1)
    # Liczba klastrów bez szumu (-1)
    unique_labels = set(dbscan_labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    n_clusters_db = len(unique_labels)

    print(f"  DBSCAN: Utworzono {n_clusters_db} grup (szum: {n_noise} punktów).")

    # 4. Redukcja wymiarów do wizualizacji (PCA)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    viz_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
    viz_df['KMeans_Labels'] = kmeans_labels
    viz_df['DBSCAN_Labels'] = dbscan_labels

    # 5. Rysowanie
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f'Klasteryzacja: {dataset_name}', fontsize=16)

    # K-Means Plot
    sns.scatterplot(data=viz_df, x='PCA1', y='PCA2', hue='KMeans_Labels', palette='viridis', s=80, ax=ax1)
    ax1.set_title(f'K-Means (k={n_clusters})')

    # DBSCAN Plot
    # Ustawiamy paletę tak, żeby szum był widoczny
    sns.scatterplot(data=viz_df, x='PCA1', y='PCA2', hue='DBSCAN_Labels', palette='deep', s=80, ax=ax2)
    ax2.set_title(f'DBSCAN (eps={eps})')

    plt.tight_layout()
    plt.show()

# --- URUCHOMIENIE ---
if __name__ == "__main__":
    # 1. Mall Customers
    data_mall, name_mall = load_data_kaggle()
    process_and_visualize(data_mall, name_mall)

    # 2. Wholesale Customers
    data_wholesale, name_wholesale = load_data_uci()
    process_and_visualize(data_wholesale, name_wholesale)