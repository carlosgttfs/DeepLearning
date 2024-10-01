import os
import pandas as pd
from math import sqrt
import numpy as np

# Função para calcular a distância euclidiana entre dois pontos
def euclidean_distance(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Função para carregar os pontos do OpenFace
def load_openface_points(openface_file):
    openface_data = pd.read_csv(openface_file)

    # Verifica se as colunas possuem o formato esperado
    column_names = openface_data.columns
    print(f"Colunas detectadas: {column_names}")  # Adicionado para inspecionar as colunas

    # Tenta detectar a estrutura correta das colunas para X e Y
    points_x = [col for col in column_names if col.startswith(' X_')]
    points_y = [col for col in column_names if col.startswith(' Y_')]

    if len(points_x) != 68 or len(points_y) != 68:
        raise ValueError(f"Número de pontos esperado (68) não encontrado no arquivo {openface_file}")

    points_x_values = openface_data[points_x].values[0]
    points_y_values = openface_data[points_y].values[0]

    return list(zip(points_x_values, points_y_values))

# Função para carregar os pontos do iBUG
def load_ibug_points(ibug_file):
    points = []
    with open(ibug_file, 'r') as f:
        for line in f:
            x, y = map(float, line.split())
            points.append((x, y))
    return points

# Função principal para comparar os pontos entre OpenFace e iBUG
def compare_points(openface_dir, ibug_dir, output_file):
    differences_summary = []

    # Iterar sobre todos os arquivos OpenFace
    for openface_file in sorted(os.listdir(openface_dir)):
        if openface_file.endswith('.csv'):
            # Nome correspondente do arquivo iBUG
            image_num = openface_file.split('_')[1]  # Extraí o número da imagem (ex: '003' de 'image_003_1.csv')
            ibug_file = f'image_{image_num}.txt'  # Assumindo que o arquivo iBUG tem esse padrão de nome

            openface_path = os.path.join(openface_dir, openface_file)
            ibug_path = os.path.join(ibug_dir, ibug_file)

            # Se o arquivo iBUG correspondente não existir, pula para o próximo
            if not os.path.exists(ibug_path):
                print(f"Arquivo iBUG não encontrado para {openface_file}, pulando...")
                continue

            # Carregar os pontos
            openface_points = load_openface_points(openface_path)
            ibug_points = load_ibug_points(ibug_path)

            # Verificar se há 68 pontos em ambos
            if len(openface_points) != 68 or len(ibug_points) != 68:
                print(f"Erro: Número de pontos não corresponde para {openface_file}")
                continue

            # Calcular as diferenças
            differences = [euclidean_distance(p1, p2) for p1, p2 in zip(openface_points, ibug_points)]

            # Salvar a média e desvio padrão das diferenças
            mean_diff = np.mean(differences)
            std_diff = np.std(differences)
            differences_summary.append((openface_file, mean_diff, std_diff))

            print(f"Comparação concluída para {openface_file}: Média={mean_diff}, Desvio Padrão={std_diff}")

    # Salvar os resultados em um arquivo de saída
    results_df = pd.DataFrame(differences_summary, columns=['Image', 'Mean Difference', 'Std Difference'])
    results_df.to_csv(output_file, index=False)
    print(f"Resultados salvos em {output_file}")

# Defina os diretórios dos arquivos OpenFace e iBUG
openface_dir = '/home/carlosguttemberg/Downloads/ibug/openface_results'  # Ajuste o caminho para seus arquivos OpenFace
ibug_dir = '/home/carlosguttemberg/Downloads/ibug/annotations'  # Ajuste o caminho para os arquivos de anotações do iBUG

# Arquivo de saída para os resultados
output_file = 'comparison_results.csv'

# Executar a comparação
compare_points(openface_dir, ibug_dir, output_file)
