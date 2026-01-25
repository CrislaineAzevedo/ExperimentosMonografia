import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import gudhi
import cv2
import os

def GetBettiCurvesFromPointCloud(J,entrada,arquivo):
    I = 2 * J
    diag_01=pd.read_csv(os.path.join(entrada, arquivo)).values
    diag_0,diag_1=[],[]
    for interval in diag_01:
        if interval[0]==0:
            diag_0.append(interval)
        else:
            diag_1.append(interval)
    Diagrams=[np.array(diag_0),np.array(diag_1)]
    BettiCurves = []
    step_x = I[1] - I[0]
    for diagram in Diagrams:
        bc = np.zeros(len(I))
        if diagram.size != 0:
            diagram_int = np.clip(
                np.ceil((diagram - I[0]) / step_x), 0, len(I)
            ).astype(int)
            for interval in diagram_int:
                bc[interval[0]:interval[1]] += 1
        BettiCurves.append(bc)
    return BettiCurves


entrada = "/content/drive/MyDrive/Monografia /saídas/Persistências concatenadas (AB+)"
saida_img = "/content/drive/MyDrive/Monografia /saídas/Curvas de Betti (AB+)/plots"
saida_csv = "/content/drive/MyDrive/Monografia /saídas/Curvas de Betti (AB+)/csvs"

raios = np.linspace(0, 10, 100)

arquivos=os.listdir(entrada)

for nome_arquivo in tqdm(arquivos, desc="Calculando curvas de Betti"):

    curva = GetBettiCurvesFromPointCloud(raios,entrada ,nome_arquivo)

    df = pd.DataFrame({
        "raio": raios,
        "betti_0": curva[0],
        "betti_1": curva[1]
    })

    nome_base = os.path.splitext(nome_arquivo)[0]
    df.to_csv(os.path.join(saida_csv, f"{nome_base}.csv"), index=False)

    plt.figure()
    plt.plot(raios, curva[0], label=r"$\beta_0$")
    plt.plot(raios, curva[1], label=r"$\beta_1$")
    plt.title(f"Curva de Betti - {nome_base}")
    plt.xlabel("Raio $r$")
    plt.ylabel("Número de Betti")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(saida_img, f"{nome_base}.png"))
    plt.close()

