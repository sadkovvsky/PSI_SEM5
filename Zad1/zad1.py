# Importy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ustawienie stylu
plt.style.use('default')
sns.set_palette("viridis")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, RocCurveDisplay)

# 1. Dane
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df['species'] = y
df['species_name'] = [target_names[i] for i in y]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Rozkład klas (cały zbiór):")
print(df['species_name'].value_counts())

# 2. EDA
print("\n=== EKSPLORACJA DANYCH ===")

# Pairplot
sns.pairplot(df, hue='species_name', diag_kind='hist', palette='viridis')
plt.suptitle("Pairplot cech zbioru Iris", y=1.02)
plt.show()

# Macierz korelacji
plt.figure(figsize=(8, 6))
sns.heatmap(df[feature_names].corr(), annot=True, cmap='coolwarm', center=0)
plt.title("Macierz korelacji cech")
plt.show()

# Boxploty - POPRAWIONE
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, feature in enumerate(feature_names):
    row, col = idx // 2, idx % 2
    sns.boxplot(data=df, x='species_name', y=feature, hue='species_name',
                ax=axes[row, col], palette='Set2', legend=False)
    axes[row, col].set_title(f'Rozkład {feature} wg gatunków')
    axes[row, col].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# 3. Modele
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipelines = {
    'KNN': Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())]),
    'SVM': Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True, random_state=42))]),
    'RandomForest': Pipeline([('clf', RandomForestClassifier(random_state=42))])
}

# Porównanie via cross_validate
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
results_cv = {}

print("\n=== WYNIKI WALIDACJI KRZYŻOWEJ ===")
for name, pipe in pipelines.items():
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring,
                            return_train_score=False, n_jobs=-1)
    results_cv[name] = {k: np.mean(v) for k, v in scores.items() if k.startswith('test_')}
    results_cv[name]['cv_std'] = np.std(scores['test_accuracy'])

    print(f"\n{name} CV results:")
    for metric, val in results_cv[name].items():
        if metric != 'cv_std':
            print(f"  {metric}: {val:.4f}")

# Wizualizacja porównania modeli - POPRAWIONE I ZABEZPIECZONE
print("\n=== GENEROWANIE WYKRESU PORÓWNANIA MODELI ===")

# Sprawdzenie czy są dane do wykreślenia
if results_cv:
    # Przygotowanie danych do wykresu
    cv_results_df = pd.DataFrame(results_cv).T

    # Usunięcie kolumny cv_std jeśli istnieje
    if 'cv_std' in cv_results_df.columns:
        metrics_to_plot = cv_results_df.drop('cv_std', axis=1)
    else:
        metrics_to_plot = cv_results_df

    print("Dane do wykresu:")
    print(metrics_to_plot)

    # Sprawdzenie czy DataFrame nie jest pusty
    if not metrics_to_plot.empty:
        # Tworzenie wykresu
        fig, ax = plt.subplots(figsize=(12, 7))

        # Upewnij się, że mamy dane liczbowe
        numeric_data = metrics_to_plot.select_dtypes(include=[np.number])

        if not numeric_data.empty:
            # Rysowanie wykresu
            numeric_data.plot(kind='bar', ax=ax, width=0.8)

            ax.set_title('Porównanie modeli - wyniki walidacji krzyżowej', fontsize=14, pad=20)
            ax.set_ylabel('Wartość metryki', fontsize=12)
            ax.set_xlabel('Model', fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

            # Dodanie wartości na słupkach
            for i, (index, row) in enumerate(numeric_data.iterrows()):
                for j, value in enumerate(row):
                    ax.text(i + j / len(row) - 0.3, value + 0.01, f'{value:.3f}',
                            ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.show()
            print("Wykres porównania modeli został wyświetlony.")
        else:
            print("Brak danych liczbowych do wykreślenia.")
    else:
        print("DataFrame metrics_to_plot jest pusty.")
else:
    print("Brak wyników w results_cv.")

# 4. Trenowanie modeli na pełnym zbiorze treningowym
print("\n=== WYNIKI NA ZBIORZE TESTOWYM ===")
fitted = {}
test_results = {}

for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    test_results[name] = accuracy
    print(f"\n{name} - test accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=target_names))
    fitted[name] = pipe

# 5. Tuning hiperparametrów
param_grid = {
    'KNN': {
        'clf__n_neighbors': list(range(1, 16)),
        'clf__weights': ['uniform', 'distance'],
        'clf__metric': ['euclidean', 'manhattan']
    },
    'SVM': {
        'clf__C': [0.1, 1, 10, 100],
        'clf__gamma': ['scale', 'auto', 0.1, 0.01],
        'clf__kernel': ['rbf', 'linear']
    },
    'RandomForest': {
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [None, 10, 20, 30],
        'clf__min_samples_split': [2, 5],
        'clf__min_samples_leaf': [1, 2]
    }
}

# Wybór najlepszego modelu
best_name = max(results_cv, key=lambda n: results_cv[n]['test_accuracy'])
print(f"\n=== TUNING HIPERPARAMETRÓW ===")
print(f"Najlepszy model wg CV: {best_name}")

grid = GridSearchCV(pipelines[best_name], param_grid[best_name], cv=cv,
                    scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
print("Najlepsze parametry:", grid.best_params_)
print(f"Najlepsze CV accuracy: {grid.best_score_:.4f}")
best_model = grid.best_estimator_

# 6. Ewaluacja finalna
print("\n=== WYNIKI KOŃCOWE ===")
y_pred_final = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred_final)

print(f"\nOstateczny model: {best_name}")
print(f"Accuracy na zbiorze testowym: {final_accuracy:.4f}")
print("\nOstateczny raport klasyfikacji:")
print(classification_report(y_test, y_pred_final, target_names=target_names))

# Macierz pomyłek
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title(f"Macierz pomyłek - {best_name} (final model)")
plt.xlabel("Przewidywane")
plt.ylabel("Rzeczywiste")
plt.tight_layout()
plt.show()

# ROC-AUC
if hasattr(best_model, "predict_proba"):
    y_proba = best_model.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    auc_scores = {}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Krzywe ROC
    for i, name in enumerate(target_names):
        auc_score = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
        auc_scores[name] = auc_score
        RocCurveDisplay.from_predictions(
            y_test_bin[:, i],
            y_proba[:, i],
            name=f"{name} (AUC = {auc_score:.3f})",
            ax=ax1
        )
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Losowy klasyfikator')
    ax1.set_title('Krzywe ROC dla poszczególnych klas')
    ax1.legend()

    # Wartości AUC
    classes = list(auc_scores.keys())
    scores = list(auc_scores.values())
    bars = ax2.bar(classes, scores, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax2.set_ylabel('AUC Score')
    ax2.set_title('Wartości AUC dla poszczególnych klas')
    ax2.set_ylim(0, 1.1)

    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{score:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    print("\nROC-AUC scores:")
    for class_name, auc_score in auc_scores.items():
        print(f"  {class_name}: {auc_score:.4f}")

# Analiza ważności cech
if best_name == 'RandomForest' and hasattr(best_model.named_steps['clf'], 'feature_importances_'):
    feature_importances = best_model.named_steps['clf'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=feature_importance_df, x='importance', y='feature', palette='viridis')
    plt.title('Wažność cech - Random Forest')
    plt.xlabel('Wažność')
    plt.tight_layout()
    plt.show()

    print("\nAnaliza ważności cech:")
    print(feature_importance_df)
