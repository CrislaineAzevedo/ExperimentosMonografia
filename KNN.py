
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

pasta_1 = "/content/drive/MyDrive/Monografia /saídas/Curvas de Betti (B+)/csvs"
pasta_2 = "/content/drive/MyDrive/Monografia /saídas/Curvas de Betti (AB+)/csvs"

X_1 = carregar_csvs(pasta_1)
X_2 = carregar_csvs(pasta_2)

y_1 = np.zeros(X_1.shape[0])
y_2 = np.ones(X_2.shape[0])

X = np.vstack((X_1, X_2))
y = np.concatenate((y_1, y_2))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


melhor_k = None
melhor_acc = 0

for k in range(1, 51):
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


