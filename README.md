
# Détection de Métiers dans des Phrases en Français avec CamemBERT

Ce projet utilise le modèle CamemBERT pour entraîner un modèle de reconnaissance d'entités nommées (NER) capable de détecter les métiers dans des phrases en français. Le projet se compose de trois étapes : génération des données, entraînement du modèle, et test des prédictions sur des phrases de test.

## Structure du Projet

- `generate_data.py` : Génère des données annotées pour l'entraînement.
- `train_model.py` : Entraîne un modèle CamemBERT pour la classification de tokens sur les données générées.
- `test_model.py` : Teste le modèle sur de nouvelles phrases pour prédire les métiers détectés.

## Prérequis

Avant de commencer, assurez-vous d'avoir installé les bibliothèques nécessaires. Vous pouvez les installer avec :

```bash
pip install pandas numpy torch transformers datasets nltk
```

## Description des Fichiers

### 1. Génération des Données (`generate_data.py`)

Ce script crée un jeu de données synthétique en français pour détecter les métiers dans des phrases. Il génère des exemples de phrases contenant des métiers (par exemple, "photographe", "développeur") ainsi que des phrases sans mention de métier, puis les étiquette pour l'entraînement.

- **Sortie** : 
  - `generated_data.csv` : contient les phrases générées avec leurs labels (1 pour la présence d'un métier, 0 sinon).
  - `annotated_data.csv` : les phrases annotées avec des labels (`B-METIER`, `I-METIER`, `O`) pour la classification de tokens.
  - `tokenized_data.csv` : les phrases tokenisées et prêtes pour l’entraînement.

### 2. Entraînement du Modèle (`train_model.py`)

Ce script utilise les données annotées pour entraîner un modèle CamemBERT de reconnaissance d'entités nommées.

- **Étapes** :
  1. Charge le tokenizer et les données tokenisées.
  2. Divise les données en ensembles d'entraînement, de validation, et de test.
  3. Définit les hyperparamètres d’entraînement.
  4. Entraîne le modèle et enregistre le modèle final dans le dossier `./trained_model`.

- **Sortie** : 
  - `./trained_model` : contient le modèle et le tokenizer entraînés.

### 3. Test du Modèle (`test_model.py`)

Ce script utilise le modèle entraîné pour faire des prédictions sur des phrases de test, affichant les métiers détectés.

- **Étapes** :
  1. Charge le modèle et le tokenizer entraînés.
  2. Tokenize les phrases de test et fait des prédictions.
  3. Affiche chaque token de la phrase avec le label prédit (`B-METIER`, `I-METIER`, `O`).

## Exécution des Scripts

Suivez ces étapes pour exécuter chaque fichier :

1. **Générer les Données** :
   ```bash
   python generate_data.py
   ```

2. **Entraîner le Modèle** :
   ```bash
   python train_model.py
   ```

3. **Tester le Modèle** :
   ```bash
   python test_model.py
   ```

## Exemple de Résultat de Test

Après l'exécution de `test_model.py`, vous verrez en sortie les tokens de chaque phrase de test avec les labels prédits :

```plaintext
Sentence: je recherche pour 2 jours un entrepreneur ou bien un photographe
Tokens and Labels:
je: O
recherche: O
pour: O
2: O
jours: O
un: O
entrepreneur: B-METIER
ou: O
bien: O
un: O
photographe: B-METIER
```

## Remarques

- Ce projet utilise CamemBERT pour la reconnaissance d’entités nommées. Assurez-vous que votre GPU est activé pour accélérer l’entraînement si possible.
- Les données générées sont artificielles et peuvent être adaptées selon les besoins.
