# Break Detection in Time Series

Un système complet de détection de breaks statistiques dans les séries temporelles, utilisant l'apprentissage automatique pour identifier les changements structurels.

**Source des données** : [Challenge CrunchDAO - Structural Break](https://hub.crunchdao.com/competitions/structural-break/resources/datasets)

## 📊 Résultats

Le système atteint typiquement :
- **AUC-ROC** : 0.70-0.75
- **Précision** : 75-80%
- **Temps de traitement** : Variable selon la taille des données

## 📁 Structure du projet

### Modules principaux

- **`config.py`** : Configuration centralisée des paramètres

- **Transformateurs** :
  - `OutlierDetectionTransformer` : Correction des erreurs de séries temporelles
  - `TimeSeriesFeatureTransformer` : Features statistiques avancées

- **Modélisation** :
  - `TimeSeriesModel` : Modélisation par série
  - `BreakDetector` : Détection des breaks

- **`pipeline`** : Orchestration du workflow complet

## ✨ Fonctionnalités

### 🔧 Ingénierie des features

- **Features temporelles** : Lag, moyennes mobiles, écarts-types
- **Détection d'outliers** : Basée sur l'erreur de prédiction
- **Features statistiques** :
  - Changements de moyenne et volatilité
  - Entropie spectrale
  - Statistiques CUSUM
  - Tests de rupture structurelle

### 🤖 Modèles implémentés

- **XGBoost** : Prédiction série temporelle par série
- **Random Forest** : Sélection de features
- **MLP** : Classification finale avec optimisation

## 🚀 Installation

```bash
# Cloner le repository
git clone git@github.com:Portfolio20224/break-detection.git
cd break-detection

# Installer les dépendances
poetry install
```



## 📈 Méthodologie

1. **Préprocessing** : Nettoyage et détection d'outliers
2. **Feature Engineering** : Extraction de caractéristiques statistiques
3. **Modélisation** : Entraînement par série temporelle
4. **Détection** : Classification des points de rupture
5. **Post-traitement** : Optimisation des prédictions

## 🔬 Approche technique

Le système combine plusieurs approches :
- **Analyse statistique** traditionnelle (CUSUM, tests de Chow)
- **Machine Learning** pour la prédiction
- **Ensemble methods** pour la robustesse
