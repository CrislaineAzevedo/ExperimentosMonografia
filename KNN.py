
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def carregar_csvs(pasta):
    vetores = []
    for arquivo in os.listdir(pasta):
        if arquivo.endswith(".csv"):
            df = pd.read_csv(os.path.join(pasta, arquivo))

            b0 = df["betti_0"].values
            b1 = df["betti_1"].values

            vetor = np.concatenate([b0, b1])
            vetores.append(vetor)

    return np.array(vetores)

pasta_a = "/content/drive/MyDrive/Monografia /saídas/Curvas de Betti (B+)/csvs"
pasta_o = "/content/drive/MyDrive/Monografia /saídas/Curvas de Betti (AB+)/csvs"

X_a = carregar_csvs(pasta_a)
X_o = carregar_csvs(pasta_o)

y_a = np.zeros(X_a.shape[0])
y_o = np.ones(X_o.shape[0])

X = np.vstack((X_a, X_o))
y = np.concatenate((y_a, y_o))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


melhor_k = None
melhor_acc = 0

print("Testando valores de k:\n")

for k in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print(f"k = {k}: acurácia = {acc:.4f}")

    if acc > melhor_acc:
        melhor_acc = acc
        melhor_k = k

print("\nMelhor resultado:")
print(f"Melhor k = {melhor_k} com acurácia = {melhor_acc:.4f}")


