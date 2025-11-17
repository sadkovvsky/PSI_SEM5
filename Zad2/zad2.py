import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)

# Ustawienie stylu wizualizacji
sns.set_style("whitegrid")

## 1. ŁADOWANIE I PRZYGOTOWANIE DANYCH
data = load_breast_cancer()
X = data.data
y = data.target

print(f"Wczytano {X.shape[0]} próbek z {X.shape[1]} cechami.")
print(f"Klasy: {data.target_names} (0: {data.target_names[0]}, 1: {data.target_names[1]})")

## 2. PODZIAŁ NA ZBIÓR TRENINGOWY I TESTOWY
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Zbiór treningowy: {X_train.shape[0]} próbek")
print(f"Zbiór testowy: {X_test.shape[0]} próbek")

## 3. SKALOWANIE DANYCH
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Dane zostały przeskalowane.")

## 4. TRENOWANIE, REGULARYZACJA I WALIDACJA KRZYŻOWA (GridSearchCV)
model = LogisticRegression(solver='liblinear', random_state=42)

# Siatka hiperparametrów do sprawdzenia
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Siła regularyzacji
    'penalty': ['l1', 'l2']       # Typ regularyzacji
}

# GridSearch z 5-krotną walidacją krzyżową, optymalizujący pod AUC
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc'
)

print("Rozpoczynam trenowanie (GridSearch)...")
grid_search.fit(X_train_scaled, y_train)

# Najlepszy model
best_model = grid_search.best_estimator_
print("\n--- Wyniki Walidacji Krzyżowej ---")
print(f"Najlepsze parametry: {grid_search.best_params_}")
print(f"Najlepszy wynik AUC (trening): {grid_search.best_score_:.4f}")

## 5. OCENA MODELU NA ZBIORZE TESTOWYM
y_pred = best_model.predict(X_test_scaled)
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

print("\n--- Ocena Modelu na Zbiorze Testowym ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall (Czułość): {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

print("\n--- Raport Klasyfikacji ---")
print(classification_report(y_test, y_pred, target_names=['Malignant (0)', 'Benign (1)']))

# Macierz pomyłek
print("\n--- Macierz Pomyłek ---")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Przewidziana etykieta')
plt.ylabel('Prawdziwa etykieta')
plt.title('Macierz Pomyłek')
plt.show()

# Krzywa ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Krzywa ROC (AUC = {roc_auc_score(y_test, y_prob):.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Model losowy')
plt.xlabel('False Positive Rate (FPF)')
plt.ylabel('True Positive Rate (TPF) - Recall')
plt.title('Krzywa ROC')
plt.legend(loc="lower right")
plt.show()