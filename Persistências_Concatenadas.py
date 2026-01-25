from tqdm import tqdm
import numpy as np
import os

pastas = [
    "/content/drive/MyDrive/Monografia /saídas/Persistências (AB+)/Persistências 1 (AB+)/csv",
    "/content/drive/MyDrive/Monografia /saídas/Persistências (AB+)/Persistências 2 (AB+)/csv",
    "/content/drive/MyDrive/Monografia /saídas/Persistências (AB+)/Persistências 3 (AB+)/csv",
    "/content/drive/MyDrive/Monografia /saídas/Persistências (AB+)/Persistências 4 (AB+)/csv",
    "/content/drive/MyDrive/Monografia /saídas/Persistências (AB+)/Persistências 5 (AB+)/csv",
    "/content/drive/MyDrive/Monografia /saídas/Persistências (AB+)/Persistências 6 (AB+)/csv",
    "/content/drive/MyDrive/Monografia /saídas/Persistências (AB+)/Persistências 7 (AB+)/csv",
    "/content/drive/MyDrive/Monografia /saídas/Persistências (AB+)/Persistências 8 (AB+)/csv",
    "/content/drive/MyDrive/Monografia /saídas/Persistências (AB+)/Persistências 9 (AB+)/csv",
]

pasta_saida = "/content/drive/MyDrive/Monografia /saídas/Persistências concatenadas (AB+)"

arquivos = sorted([f for f in os.listdir(pastas[0]) if f.endswith(".csv")])

for nome_arquivo in tqdm(arquivos):
    todas = []

    for pasta in pastas:
        caminho = os.path.join(pasta, nome_arquivo)

        dados = np.genfromtxt(caminho, delimiter=",", skip_header=1)
        todas.append(dados)

    persistencia_final = np.vstack(todas)

    saida = os.path.join(pasta_saida, nome_arquivo)
    np.savetxt(
        saida,
        persistencia_final,
        delimiter=",",
        header="nascimento,morte",
        comments=""
    )
