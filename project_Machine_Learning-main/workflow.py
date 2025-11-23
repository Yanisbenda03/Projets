# ----------------------
# Imports communs
# ----------------------
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ----------------------
# Helpers communs
# ----------------------
def _impute_and_scale(X: pd.DataFrame, strategy: str) -> tuple:
    imputer = SimpleImputer(strategy=strategy)
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    return X_scaled, imputer, scaler

# ----------------------
# Diabete Dataset
# ----------------------
def load_diabetes_df(diabetes_path) -> pd.DataFrame:
    return pd.read_csv(diabetes_path)

def clean_diabetes(diabetes_path: str, strategy="mean") :
    df = load_diabetes_df(diabetes_path)
    y = df["Diabetes_binary"].to_numpy()
    X = df.drop(columns=["Diabetes_binary"]).apply(pd.to_numeric, errors="coerce")
    X_scaled, imputer, scaler = _impute_and_scale(X, strategy)
    return {
        "X": X_scaled,
        "y": y,
        "feature_names": X.columns.tolist(),
        "imputer": imputer,
        "scaler": scaler,
    }

# ----------------------
# Spambase Dataset
# ----------------------
def _read_spambase_names(spambase_names_path):
    try:
        with open(spambase_names_path, encoding="utf-8", errors="ignore") as f:
            cols = []
            for line in f:
                line = line.strip()
                if not line or line.startswith("|"):
                    continue
                # lignes du style "word_freq_make: continuous."
                if ":" in line:
                    cols.append(line.split(":")[0].strip())
            if "class" not in [c.lower() for c in cols]:
                cols.append("class")
            return cols
    except FileNotFoundError:
        # 57 features + 1 target selon UCI
        return [f"feat_{i}" for i in range(57)] + ["class"]

def load_spambase_df(spambase_data_path, spambase_names_path) -> pd.DataFrame:
    cols = _read_spambase_names(spambase_names_path)
    return pd.read_csv(spambase_data_path, header=None, names=cols)

def clean_spambase(spambase_data_path,spambase_names_path, strategy="median") :
    df = load_spambase_df(spambase_data_path, spambase_names_path)
    y = df["class"].to_numpy()
    X = df.drop(columns=["class"]).apply(pd.to_numeric, errors="coerce")
    X_scaled, imputer, scaler = _impute_and_scale(X, strategy)
    return {
        "X": X_scaled,
        "y": y,
        "feature_names": X.columns.tolist(),
        "imputer": imputer,
        "scaler": scaler,
    }

# ----------------------
# Split train/test
# ----------------------

def split_dataset(data_dict: dict, test_size: float = 0.2, random_state: int = 42): #ajout d'un noyau d'aléatoire pour assurer la reproductivité | 20% du dataset en test
    X_train, X_test, y_train, y_test = train_test_split(
        data_dict["X"],
        data_dict["y"],
        test_size=test_size,
        random_state=random_state,
        stratify=data_dict["y"] if len(set(data_dict["y"])) > 1 else None  #conserve la même proportion de classes entre train/test
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

# ----------------------
# Equilibrage training set
# ----------------------

def balance_training_set(X_train, y_train, random_state=42):
   
    sm = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

    return X_train_bal, y_train_bal
# ----------------------    
# Select features pipeline
# ----------------------

def select_features_1(X_train, y_train, X_test, k):  #K: nombre de features à conserver, la fonction de score ici pour f_classif repose sur l'analyse de la variance (corrélation entre la feature et la cible)
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    selected_features = selector.get_support(indices=True)
    
    return X_train_selected, X_test_selected, selected_features

# ----------------------
# Fonctions méthode: réseau de neurones (MLP)
# ----------------------

#création MLP à 2 hidden layers 
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    
class MLP_2(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

#fonction entraînement NN
  
def train_mlp(neural_network, X_train, y_train, X_test, y_test, num_epochs=20, lr=1e-3):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)
    model = neural_network(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    losses=[]

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test)
        y_pred_labels = (y_pred_test > 0.5).float()
        accuracy = (y_pred_labels == y_test).float().mean().item()

    print(f"Test accuracy: {accuracy:.4f}")
    return model, accuracy, losses

# ----------------------
# Fonctions méthode: Régression Logistique
# ----------------------

def train_logistic_regression(X_train, y_train, X_test, y_test, max_iter=1000):
   
    # 1) Création du modèle
    model = LogisticRegression(max_iter=max_iter,class_weight='balanced')

    # 2) Entraînement
    model.fit(X_train, y_train)

    # 3) Prédiction 
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # 4) Accuracy 
    accuracy = accuracy_score(y_test, y_pred)

  

    print(f"Test accuracy (LogReg): {accuracy:.4f}")

    return model, accuracy

# ----------------------
# Fonctions méthode: Random Forest 
# ----------------------

def train_random_forest(
    X_train, y_train,
    X_test, y_test,
    n_estimators=100,
    max_depth=None,
    random_state=42
):
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Random Forest (n_estimators={n_estimators}, max_depth={max_depth}) - Accuracy test : {acc:.4f}")


    return model, acc

# ----------------------
# Fonctions méthode: KNN
# ----------------------

def knn_compute_scores(K_values, X_train, y_train, n_splits=10):
  
    K_max = len(K_values)
    acc_values = np.zeros((K_max, n_splits))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for i, k in enumerate(K_values):
        knn = KNeighborsClassifier(n_neighbors=k)
        acc_values[i, :] = cross_val_score(knn, X_train, y_train, cv=cv)

    # Best K
    mean_acc = acc_values.mean(axis=1)
    best_index = np.argmax(mean_acc)
    best_k = K_values[best_index]
    best_acc = mean_acc[best_index]

    return best_k, best_acc, acc_values


def train_knn(X_train, y_train, X_test, y_test, n_neighbors):
   
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc


# ----------------------
# Fonctions d'évaluation
# ----------------------

def evaluate_classification(y_train, y_train_pred, y_test, y_test_pred, plot_confusion=True):

    y_train_pred_bin = (y_train_pred > 0.5).astype(int) if y_train_pred.ndim > 1 else y_train_pred
    y_test_pred_bin = (y_test_pred > 0.5).astype(int) if y_test_pred.ndim > 1 else y_test_pred
    
    metrics = {
        "train_accuracy": accuracy_score(y_train, y_train_pred_bin),
        "test_accuracy": accuracy_score(y_test, y_test_pred_bin),
        "train_f1": f1_score(y_train, y_train_pred_bin),
        "test_f1": f1_score(y_test, y_test_pred_bin),
        "train_precision": precision_score(y_train, y_train_pred_bin),
        "test_precision": precision_score(y_test, y_test_pred_bin),
        "train_recall": recall_score(y_train, y_train_pred_bin),
        "test_recall": recall_score(y_test, y_test_pred_bin),
    }
    
    # Matrice de confusion test
    if plot_confusion:
        cm = confusion_matrix(y_test, y_test_pred_bin)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        plt.title("Matrice de confusion - Test")
        plt.show()
    
    return metrics

# ----------------------
# Tests (importation datasets)
# ----------------------
if __name__ == "__main__":
    diabetes_path = "Diabete/diabetes_binary_health_indicators_BRFSS2015.csv"
    spambase_data_path = "spambase/spambase.data"
    spambase_names_path = "spambase/spambase.names"
    d = clean_diabetes(diabetes_path)
    print("Diabetes:", d["X"].shape, len(d["feature_names"]), "features")
    s = clean_spambase(spambase_data_path)
    print("Spambase:", s["X"].shape, len(s["feature_names"]), "features")


