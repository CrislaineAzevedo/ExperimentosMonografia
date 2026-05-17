import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from gudhi.representations import Landscape

entrada = "/content/drive/MyDrive/Crislaine/saídas/Persistências concatenadas (AB+)"
saida_img = "/content/drive/MyDrive/Crislaine/saídas/Landscapes (AB+)/plots"
saida_csv = "/content/drive/MyDrive/Crislaine/saídas/Landscapes (AB+)/csvs"

num_landscape = 30
pontos = 1000

todos_os_arquivos = os.listdir(entrada)

diagrama_arquivos = sorted(os.listdir(entrada))

for arquivo_nome in diagrama_arquivos:
    caminho_arquivo = os.path.join(entrada, arquivo_nome)

    diagrama_bd = pd.read_csv(caminho_arquivo).values

    # Tratar valores infinitos
    filtra_val_finitos = np.isfinite(diagrama_bd).all(axis=1)
    valores_finitos = diagrama_bd[filtra_val_finitos] #Seleciona apenas os pares nascimento e morte com valores finitos

    if valores_finitos.size > 0:
        max_finite = np.max(valores_finitos)
        diagrama_bd[~filtra_val_finitos] = max_finite * 2  # Substitui infinitos

    # Filtrar pontos válidos (death > birth)
    pontos_validos = (diagrama_bd[:, 1] > diagrama_bd[:, 0])
    diagrama_bd = diagrama_bd[pontos_validos]


    # Calcular Persistence Landscapes
    landscapes = Landscape(num_landscapes=num_landscape, resolution=pontos) #Quantas funções landscapes de nível k serão calculadas e quantos pontos iremos usar para discretizar o eixo x
    L = landscapes.fit_transform([diagrama_bd])


    L = L.reshape(num_landscape, pontos)

    plt.figure(figsize=(10, 5))
    for k in range(num_landscape):
        plt.plot(np.linspace(np.min(diagrama_bd),np.max(diagrama_bd),pontos),L[k, :], label=f'λ{k+1}')

    output_img = os.path.join(saida_img, f"{os.path.splitext(arquivo_nome)[0]}.png")
    plt.savefig(output_img, dpi=150, bbox_inches='tight')
    plt.close()

    df_landscapes = pd.DataFrame(L.T, columns=[f"lambda_{i+1}" for i in range(num_landscape)])
    df_landscapes.insert(0, "x", np.linspace(np.min(diagrama_bd),np.max(diagrama_bd),pontos))

    output_csv = os.path.join(saida_csv, f"{os.path.splitext(arquivo_nome)[0]}.csv")
    df_landscapes.to_csv(output_csv, index=False)

print('\nProcessamento concluído!')