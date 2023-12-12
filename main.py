import cv2
import os
import glob
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim



def get_diretorios():
    diretorio_main = os.path.dirname(os.path.abspath(__file__))

    diretorio_entrada = diretorio_main + "/png"
    diretorio_saida = diretorio_main + "/output"
    return diretorio_entrada, diretorio_saida    

def get_diretorio_saida_img_atual(diretorio_saida, nome_sem_extensao):
    diretorio_saida_img_atual = diretorio_saida + '/' + nome_sem_extensao
    validar_diretorio_saida(diretorio_saida_img_atual)
    return diretorio_saida_img_atual

def get_arquivos(diretorio_entrada):
    return os.listdir(diretorio_entrada)

def validar_diretorio_saida(diretorio_saida_img_atual):
    if os.path.exists(diretorio_saida_img_atual):
        # Remove o diretório caso ele exista, para criar corretamente o GIF depois
        shutil.rmtree(diretorio_saida_img_atual)
    os.makedirs(diretorio_saida_img_atual)

def ler_img(diretorio_entrada, diretorio_saida_img_atual, arquivo):
    caminho_imagem = os.path.join(diretorio_entrada, arquivo)
    img = cv2.imread(caminho_imagem, cv2.IMREAD_UNCHANGED)
    # Cria uma cópia da imagem original no diretório de saída para gerar o GIF na ordem correta
    cv2.imwrite(os.path.join(diretorio_saida_img_atual, f'0_{arquivo}'), img)
    # Altera a ordem dos canais de cores
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Trnsforma a matriz da imagem para duas dimensões
    img_redimensionada = img.reshape((-1,3))
    # Converte para float32 os valores da matriz
    img_redimensionada = np.float32(img_redimensionada)
    return img, img_redimensionada

def get_stop_criteria():
    #critério de parada
    #TERM_CRITERIA_EPS - parada por atingir a precisão epsilon 1.0
    #TermCriteria_MAX_ITER - parada por atingir o número máximo de iterações 10
    return (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 10, 1.0)

def processar(k,index, arquivo, diretorio_saida_img_atual, img_redimensionada, qtde_cores, tamanho_imgs):
    print(f'Processando k = {k} para a imagem {arquivo}')
    ret, rotulo, centroide = cv2.kmeans(img_redimensionada, k, None, get_stop_criteria(), 10, cv2.KMEANS_RANDOM_CENTERS)
    centroide = np.uint8(centroide)
    res = centroide[rotulo.flatten()]
    qtde_cores.append(len(np.unique(res, axis=0)))
    res = res.reshape((img.shape))
    # Converte novamente para BGR para salvar a imagem
    output = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(diretorio_saida_img_atual, f'{index}_k{k}_{arquivo}'), output)
    # Calcula o tamanho da imagem em KB
    tamanho_img_kb = os.path.getsize(os.path.join(diretorio_saida_img_atual, f'{index}_k{k}_{arquivo}'))/1024
    tamanho_imgs.append(f'{tamanho_img_kb:.2f}')
    return res
        
def gera_gif(diretorio_saida_img_atual, nome_sem_extensao, arquivo):
    print(f'Gerando GIF para a imagem {arquivo}')  
    frames = [Image.open(imagem) for imagem in sorted(glob.glob(f"{diretorio_saida_img_atual}/*.png"))]
    primeiro_frame = frames[0]
    primeiro_frame.save(f"{diretorio_saida_img_atual}/{nome_sem_extensao}.gif", format="GIF", 
                        append_images=frames, save_all=True, duration=500, loop=0)

def cria_figura():
    # Cria uma figura com tamanho aproximado a 3840x2160 e 600 DPI
    return plt.figure(figsize=(3.6, 6.4), dpi=600) 

def add_img_figura(index, imagens, titulos, fig):
    ax = fig.add_subplot(4, 2, index)
    ax.imshow(imagens)
    ax.set_title(titulos)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def salva_figura(diretorio_saida_img_atual, arquivo):
    plt.subplots_adjust(left=0.03, bottom=0.05, right=0.99, top=0.97, hspace=0.05, wspace=0.1)
    plt.savefig(os.path.join(diretorio_saida_img_atual, f'out_{arquivo}'),  bbox_inches="tight")
    plt.clf()  # Limpar a figura atual para criar uma nova na próxima iteração

def salva_csv(k, tamanho_imgs, qtde_cores, indice_similaridade, diretorio_saida_img_atual, nome_sem_extensao):
    dados = {'K': k, 'Tamanho da imagem em KB': tamanho_imgs, 'Quantidade de cores': qtde_cores, 'Índice de similaridade': indice_similaridade}
    df = pd.DataFrame(dados)
    df.to_csv(os.path.join(diretorio_saida_img_atual, f'dados_{nome_sem_extensao}.csv'), index=False)

def calcula_similaridade(img, generated_img):
    # Calcula o Índice de Similaridade Estrutural para cada um dos canais de cor
    ssim_r = ssim(img[:, :, 0], generated_img[:, :, 0])
    ssim_g = ssim(img[:, :, 1], generated_img[:, :, 1])
    ssim_b = ssim(img[:, :, 2], generated_img[:, :, 2])

    # Retorna o indíce médio de similaridade estrutural entre os canais de cor
    return np.mean([ssim_r, ssim_g, ssim_b])


if __name__ == "__main__":
    diretorio_entrada, diretorio_saida  = get_diretorios()
    lista_arquivos = get_arquivos(diretorio_entrada)

    for arquivo in lista_arquivos:
        if arquivo.endswith(".png"):
            nome_sem_extensao = arquivo.split('.')[0]
            diretorio_saida_img_atual = get_diretorio_saida_img_atual(diretorio_saida, nome_sem_extensao)
            img, img_redimensionada = ler_img(diretorio_entrada, diretorio_saida_img_atual, arquivo)
            tamanho_img_kb = os.path.getsize(os.path.join(diretorio_entrada, arquivo))/1024
            tamanho_imgs = [f'{tamanho_img_kb:.2f}']
            qtde_cores = [len(np.unique(img_redimensionada, axis=0))]
            indice_similaridade = [f'{calcula_similaridade(img, img):.2f}']
            imagens = [img]
            titulos = ["Original"]
            k = [0, 12, 8, 6, 5, 4, 3, 2]
            fig = cria_figura()
            for i in range(8):
                if i > 0 :
                    generated_img = processar(k[i], i, arquivo, diretorio_saida_img_atual, img_redimensionada, qtde_cores, tamanho_imgs)
                    imagens.append(generated_img)
                    titulos.append("K = " + str(k[i]))
                    indice_similaridade.append(f'{calcula_similaridade(img, generated_img):.2f}')

                ax = add_img_figura(i+1, imagens[i], titulos[i], fig)

            gera_gif(diretorio_saida_img_atual, nome_sem_extensao, arquivo)
            salva_figura(diretorio_saida_img_atual, arquivo)
            salva_csv(k, tamanho_imgs, qtde_cores, indice_similaridade, diretorio_saida_img_atual, nome_sem_extensao)
            