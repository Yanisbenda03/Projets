## Projet ML – Détection du diabète & du spam

Ce dépôt rassemble deux études de classification supervisée menées dans le cadre de l’UE **TAF MCE – Machine Learning** :

- `diabete_notebook.ipynb` : prédiction du risque de diabète à partir du jeu de données **BRFSS 2015 (Diabetes Health Indicators)**.
- `spambase_notebook-2.ipynb` : détection automatique de spam sur le jeu de données **UCI Spambase**.

L’ensemble des fonctions de prétraitement et d’entraînement est centralisé dans  `workflow.py`.

---

## Jeux de données

| Dataset | Taille | Caractéristiques | Particularités |
| --- | --- | --- | --- |
| `Diabete/diabetes_binary_health_indicators_BRFSS2015.csv` | 253 680 lignes × 21 features | Indicateurs de santé normalisés | **Fort déséquilibre (≈14 % de positifs)** |
| `spambase/spambase.data` + `spambase.names` | 4 601 mails × 57 features | Fréquences de mots / caractères | Données déjà équilibrées |

Les fonctions `clean_diabetes` et `clean_spambase` (voir `workflow.py`) assurent l’imputation, la normalisation et le renommage des variables avant la séparation train/test.

---

## Organisation du dépôt

```
Project_ML/
├─ base.py & workflow.py      # Prétraitement, modèles, métriques
├─ diabete_notebook.ipynb     # Analyse + résampling + comparaison des modèles
├─ spambase_notebook-2.ipynb  # Analyse spam + comparaison des modèles
├─ Diabete/                   # Données brutes BRFSS
└─ spambase/                  # Données brutes Spambase
```

---

## Environnement

- Python ≥ 3.10
- Dépendances principales : `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `torch`

Installation rapide :

```bash
python -m venv .venv
.venv\Scripts\activate        # ou source .venv/bin/activate
pip install -r requirements.txt  # si disponible
```

---

## Reproduire les expériences

1. Placer les jeux de données dans `Diabete/` et `spambase/` (déjà présent dans ce dépôt).
2. Activer l’environnement virtuel puis lancer Jupyter :
   ```bash
   jupyter notebook
   ```
3. Ouvrir et exécuter les cellules des notebooks dans l’ordre :
   - `diabete_notebook.ipynb`
   - `spambase_notebook-2.ipynb`

Les notebooks s’appuient uniquement sur les fonctions exposées dans `base.py`.  
Pour rejouer un entraînement en script Python, importer `workflow.py` et réutiliser `split_dataset`, `balance_training_set`, `train_mlp`, etc.

---

## Résultats principaux

### 1. Détection du diabète (données déséquilibrées)

| Modèle (après résampling) | Accuracy test | F1-score test | Recall test | Commentaire |
| --- | --- | --- | --- | --- |
| **MLP 1** (2 couches) | 0.726 | 0.439 | **0.767** | Recall maximal, compromis accuracy/F1 correct |
| **MLP 2** (3 couches) | 0.735 | 0.437 | 0.738 | Similaire au MLP 1 |
| **Régression logistique** | 0.729 | 0.437 | 0.755 | Modèle linéaire stable, peu sensible au résampling |
| KNN (k=2) | **0.823** | 0.319 | 0.297 | Accuracy élevée mais recall insuffisant |
| Random Forest | **0.841** | 0.318 | 0.266 | Surapprentissage, recall trop faible |

**À retenir :**
- Avant résampling, les réseaux de neurones affichaient un recall quasi nul (<2 %). L’oversampling (SMOTE-like) est indispensable.
- Pour un usage médical, le recall prime sur l’accuracy → privilégier MLP 1 ou la régression logistique.

### 2. Détection de spam (données équilibrées)

| Modèle | Accuracy test | F1-score test | Recall test | Commentaire |
| --- | --- | --- | --- | --- |
| **Random Forest** | **0.941** | **0.924** | 0.906 | Meilleur modèle mais léger surapprentissage train (99.9 %) |
| MLP 2 (3 couches) | 0.923 | – | – | Légère hausse vs MLP 1 |
| MLP 1 (2 couches) | 0.916 | – | – | Convergence stable en 4000 epochs |
| Régression logistique | 0.908 | 0.882 | 0.873 | Excellent compromis precision/recall sans surapprentissage |
| KNN (k=11) | 0.895 | 0.859 | 0.813 | Precision élevée, recall plus faible |

**À retenir :**
- Tous les modèles dépassent 89 % d’accuracy : le dataset est bien séparé.
- Random Forest domine (F1 = 0.924) mais nécessite un peu de régularisation.
- Pour l’interprétabilité, la régression logistique reste une excellente option.


---

## Auteurs

- **Vincent Brunet**
- **Yanis Bendaguir**
- **Matteo Jocal**

Projet réalisé en novembre 2025 – IMT Atlantique, UE Machine Learning.
