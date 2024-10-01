import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Definir caminho da pasta contendo os arquivos CSV
pasta = '/home/carlosguttemberg/Downloads/Eu-001/long_format'
arquivo_resultado = '/home/carlosguttemberg/Downloads/Eu-001/resultados_comparacao/resultados_comparacao.csv'
arquivo_progresso = '/home/carlosguttemberg/Downloads/Eu-001/resultados_comparacao/progresso_comparacao.csv'


# Função para carregar as coordenadas de um arquivo CSV, limitando a 68 pontos
def carregar_coordenadas(arquivo, num_pontos=68):
    df = pd.read_csv(arquivo, delimiter=',', skiprows=1, names=['x', 'y'], dtype={'x': float, 'y': float})
    return df.to_numpy()[:num_pontos]  # Seleciona os primeiros 'num_pontos'


# Função para calcular a distância Euclidiana entre dois conjuntos de coordenadas
def calcular_distancia(face1, face2):
    return np.sqrt(np.sum((face1 - face2) ** 2, axis=1))

# Carregar a lista de arquivos CSV da pasta
arquivos = sorted([os.path.join(pasta, f) for f in os.listdir(pasta) if f.endswith('.csv')])

# Função para salvar o progresso e os resultados
def salvar_resultados(resultados_parciais, arquivo_destino, cabecalho=True):
    df_parcial = pd.DataFrame(resultados_parciais, columns=['Arquivo 1', 'Arquivo 2', 'Distância Média'])
    df_parcial.to_csv(arquivo_destino, mode='a', header=cabecalho, index=False)

# Verificar se o arquivo de progresso existe, caso exista continuar de onde parou
if os.path.exists(arquivo_progresso):
    progresso_df = pd.read_csv(arquivo_progresso)
    comparacoes_feitas = set(tuple(x) for x in progresso_df[['Arquivo 1', 'Arquivo 2']].values)
    print(f'Continuando a partir de {len(comparacoes_feitas)} comparações já realizadas.')
else:
    comparacoes_feitas = set()
    # Inicializar o arquivo de resultados se não existir
    with open(arquivo_resultado, 'w') as f:
        f.write('Arquivo 1,Arquivo 2,Distância Média\n')

resultados = []
comparacoes_realizadas = 0
total_comparacoes = len(arquivos) * (len(arquivos) - 1) // 2  # Cálculo de combinações possíveis (C(n, 2))

# Barra de progresso usando tqdm
with tqdm(total=total_comparacoes, desc="Comparações", unit="comparação") as pbar:
    # Comparar todos os arquivos entre si
    for i in range(len(arquivos)):
        face1 = carregar_coordenadas(arquivos[i])
        for j in range(i + 1, len(arquivos)):
            comparacao_atual = (arquivos[i], arquivos[j])
            
            # Pular comparações já realizadas
            if comparacao_atual in comparacoes_feitas:
                continue

            face2 = carregar_coordenadas(arquivos[j])
            distancia_media = np.mean(calcular_distancia(face1, face2))
            resultados.append([arquivos[i], arquivos[j], distancia_media])
            comparacoes_realizadas += 1

            # Atualizar a barra de progresso
            pbar.update(1)

            # Salvar progresso a cada 10.000 comparações
            if comparacoes_realizadas % 10000 == 0:
                print(f'Salvando progresso após {comparacoes_realizadas} comparações...')
                salvar_resultados(resultados, arquivo_resultado, cabecalho=False)
                salvar_resultados(resultados, arquivo_progresso, cabecalho=False)
                resultados = []  # Limpar os resultados parciais após salvar

# Salvar os resultados finais
if resultados:
    salvar_resultados(resultados, arquivo_resultado, cabecalho=False)
    salvar_resultados(resultados, arquivo_progresso, cabecalho=False)

print(f'Comparações concluídas: {comparacoes_realizadas}')

