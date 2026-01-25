from gudhi.sklearn.rips_persistence import RipsPersistence
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import gudhi
import cv2
import os

pastas_saida = [
    "C:\Users\crisl\Monografia\saídas\persistências (A+)\persistência 1(A+)",
    "C:\Users\crisl\Monografia\saídas\persistências (A+)\persistência 2(A+)",
    "C:\Users\crisl\Monografia\saídas\persistências (A+)\persistência 3(A+)",
    "C:\Users\crisl\Monografia\saídas\persistências (A+)\persistência 4(A+)",
    "C:\Users\crisl\Monografia\saídas\persistências (A+)\persistência 5(A+)",
    "C:\Users\crisl\Monografia\saídas\persistências (A+)\persistência 6(A+)",
    "C:\Users\crisl\Monografia\saídas\persistências (A+)\persistência 7(A+)",
    "C:\Users\crisl\Monografia\saídas\persistências (A+)\persistência 8(A+)",
    "C:\Users\crisl\Monografia\saídas\persistências (A+)\persistência 9(A+)"
]

pasta_entrada = r"C:\Users\crisl\Monografia\digitais\A+\A+"

limiar = 150

imagens = os.listdir(pasta_entrada)
imagens = imagens[:500]

for idx, pasta_saida in enumerate(pastas_saida, start=1):

    saida_img = os.path.join(pasta_saida, "plots")
    saida_csv = os.path.join(pasta_saida, "csv")

    os.makedirs(saida_img, exist_ok=True)
    os.makedirs(saida_csv, exist_ok=True)

    for nome_arquivo in tqdm(imagens, desc=f"Rodada {idx} - Gerando persistências"):

        caminho_img = os.path.join(pasta_entrada, nome_arquivo)
        img = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)

        y, x = np.where(img < limiar)
        total_pontos = len(x)


        num_amostra = int(0.8 * total_pontos)
        amostra = np.random.choice(total_pontos, num_amostra, replace=False)
        x_amostra = x[amostra]
        y_amostra = y[amostra]

        pontos = np.column_stack((x_amostra, y_amostra))

        rips = RipsPersistence(
            homology_dimensions=(0, 1),
            n_jobs=-1,
            homology_coeff_field=2,
            threshold=20
        )
        persistence = rips.fit_transform([pontos])

        nome_base = os.path.splitext(nome_arquivo)[0]
        caminho_png = os.path.join(saida_img, nome_base + "_diagrama.png")
        caminho_csv = os.path.join(saida_csv, nome_base + "_persistencia.csv")

        gudhi.plot_persistence_diagram(persistence[0])
        plt.title(f"Diagrama de Persistência - {nome_base}")
        plt.savefig(caminho_png, dpi=300)
        plt.close()

        dados = np.concatenate((persistence[0][0], persistence[0][1]), axis=0)
        np.savetxt(caminho_csv, dados, delimiter=",", header="nascimento,morte", comments='')
