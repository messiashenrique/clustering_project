import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

###############################################################################
############################### PRÉ-PROCESSAMENTO #############################
###############################################################################
# Função para ler o arquivo ARFF e retornar o dataframe e os metadados
def read_arff(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    return df, meta

# Função para préprocessar o dataframe 
def preprocessing(df, meta):
    # Decodificar colunas de bytes para strings
    for column in df.columns:
        if df[column].dtype == object or df[column].dtype.name == 'category':
            df[column] = df[column].apply(lambda x: x.decode('utf-8').strip().lower() if isinstance(x, bytes) else x)

    # Função para preencher valores faltantes
    for column in df.columns:
        if meta[column][0] == 'numeric':
            df[column] = df[column].fillna(df[column].mean())
        else:
            df[column] = df[column].replace('?', pd.NA)
            df[column] = df[column].fillna(df[column].mode()[0])
    
    # Transformar colunas categóricas em números
    for column in df.columns:
        if df[column].dtype == object or df[column].dtype.name == 'category':
            label_encoder = LabelEncoder()
            df[column] = label_encoder.fit_transform(df[column])
    
    # Verificar se deseja normalizar as colunas numéricas usando o padrão MinMaxScaler [-1, 1]
    if input("\nDeseja normalizar os dados? (s/N): ") == 's':
        scaler = MinMaxScaler(feature_range=(-1, 1))
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # df.to_csv('teste.csv', index=False)
    return df
      
# Função para extrair os dados e rótulos verdadeiros do dataframe
def extract_data_and_labels_from_dataframe(df):
    X = df.iloc[:, :-1].values
    labels_true = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    return X, labels_true


###############################################################################
################################ ALGORITMOS ###################################
# Definir o intervalo de clusters a serem testados pelo método da silhueta
range_n_clusters = list(range(2, 10))###############################################################################

# K-Means
def run_kmeans(X, k=None):
    silhouette_avg = []
    start = time.time()
    if k is None:
        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            silhouette_avg.append(silhouette_score(X, cluster_labels))
        k = range_n_clusters[silhouette_avg.index(max(silhouette_avg))]
    kmeans = KMeans(n_clusters=k, random_state=0)
    y_kmeans = kmeans.fit_predict(X)
    end = time.time()
    time_kmeans = end - start

    return y_kmeans, time_kmeans, silhouette_avg

# Single-Link
def run_single_link(X, k=None):
    silhouette_avg = []
    start = time.time()
    if k is None:
        for n_clusters in range_n_clusters:
            single_link = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
            cluster_labels = single_link.fit_predict(X)
            silhouette_avg.append(silhouette_score(X, cluster_labels))
        k = range_n_clusters[silhouette_avg.index(max(silhouette_avg))]
    single_link = AgglomerativeClustering(n_clusters=k, linkage='single')
    y_single_link = single_link.fit_predict(X)
    end = time.time()
    time_single_link = end - start

    return y_single_link, time_single_link, silhouette_avg

# Spectral Clustering
def run_spectral(X, k=None):
    silhouette_avg = []
    start = time.time()
    if k is None:
        for n_clusters in range_n_clusters:
            spectral = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', affinity="nearest_neighbors", random_state=42)
            cluster_labels = spectral.fit_predict(X)
            silhouette_avg.append(silhouette_score(X, cluster_labels))
        k = range_n_clusters[silhouette_avg.index(max(silhouette_avg))]
    spectral = SpectralClustering(n_clusters=k, assign_labels='discretize', affinity="nearest_neighbors", random_state=42)
    y_spectral = spectral.fit_predict(X)
    end = time.time()
    time_spectral = end - start

    return y_spectral, time_spectral, silhouette_avg

###############################################################################
############################## PROGRAMA PRINCIPAL #############################
###############################################################################
bases = ["iris", "diabetes", "labor", "obesity"]    
print("\nPor favor, escolha uma das 4 bases de dados abaixo:")
for i in range(len(bases)):
    print(f"\033[32m{i+1}. {bases[i]}\033[0m")

indice = input("\nDigite um número entre 1 e 4 para escolher: ")
while True:
    if (indice.isdigit() and 1 <= int(indice) <= 4):
        break
    else:
        indice = input("\nPor favor, digite um número entre 1 e 4: ")
        continue            
    
database = bases[int(indice)-1] + ".arff"  # Caminho para o arquivo ARFF
# Imprime na tela a base escolhida
print(f"\nDatabase escolhida: \033[31m{database}\033[0m")
# Definir o número de clusters com base no número de classificações de cada base
k = [3, 2, 2, 7][int(indice)-1]


df, meta = read_arff(database)

df = preprocessing(df, meta)
X, labels_true = extract_data_and_labels_from_dataframe(df)


# Verifica se o clusters serão calculados automaticamente através do método da silhueta 
clusters_automatically = False
if input("\nDeseja que o número de clusters seja calculado automaticamente? (s/N): ") == 's':
    k = None
    clusters_automatically = True


y_kmeans, time_kmeans, avg_kmeans = run_kmeans(X, k)
y_single_link, time_single_link, avg_single_link = run_single_link(X, k)
y_spectral, time_spectral, avg_spectral = run_spectral(X, k)

# Mostrando o número de pontos em cada cluster
unique_kmeans, counts_kmeans = np.unique(y_kmeans, return_counts=True)
unique_single_link, counts_single_link = np.unique(y_single_link, return_counts=True)
unique_spectral, counts_spectral = np.unique(y_spectral, return_counts=True)

# Mostrando um sumário dos clusters
print('\n\033[47m================================== AGRUPAMENTOS ==================================\033[0m')
# Mostrando o número de pontos em cada cluster
def print_cluster_counts(y, algorithm_name, time_execution):
    unique, counts = np.unique(y, return_counts=True)
    print(f'\n\033[31m {algorithm_name}\033[0m: (Time: {time_execution:.4f} segundos)')
    for cluster, count in zip(unique, counts):
        print(f'   Cluster {cluster+1}: \033[34m{count} pontos\033[0m')

print_cluster_counts(y_kmeans, 'K-Means', time_kmeans)
print_cluster_counts(y_single_link, 'Single-Link', time_single_link)
print_cluster_counts(y_spectral, 'Spectral Clustering', time_spectral)


###############################################################################
############################ MEDIDAS DE VALIDAÇÃO #############################
###############################################################################
# Calculando o Coeficiente da Silhueta
silhouette_kmeans = silhouette_score(X, y_kmeans)
silhouette_single_link = silhouette_score(X, y_single_link)
silhouette_spectral = silhouette_score(X, y_spectral)

# Calculando o Adjusted Rand Index (ARI)
ari_kmeans = adjusted_rand_score(labels_true, y_kmeans)
ari_single_link = adjusted_rand_score(labels_true, y_single_link)
ari_spectral = adjusted_rand_score(labels_true, y_spectral)

print('\n\033[47m============================== MEDIDAS DE VALIDAÇÃO ==============================\033[0m')
print('\n\033[31mCoeficiente da Silhueta - Medida de validação Interna\033[0m:')
print(f'   K-Means: \033[32m{silhouette_kmeans:.4f}\033[0m')
print(f'   Single-Link: \033[32m{silhouette_single_link:.4f}\033[0m')
print(f'   Spectral Clustering: \033[32m{silhouette_spectral:.4f}\033[0m')

print('\n\033[31mAdjusted Rand Index (ARI) - Medida de validação Externa\033[0m:')
print(f'   K-Means: \033[32m{ari_kmeans:.4f}\033[0m')
print(f'   Single-Link: \033[32m{ari_single_link:.4f}\033[0m')
print(f'   Spectral Clustering: \033[32m{ari_spectral:.4f}\033[0m')


###############################################################################
############################ PLOTANDO OS GRÁFICOS #############################
###############################################################################

# Função para plotar os clusters
show_plots = input("\nDigite \"ok\" para ver os gráficos do agrupamento ou tecle enter para encerrar. ")

if show_plots.lower() == "ok":

    if clusters_automatically:
        # Função para plotar o gráfico do cálculo do número ideal de clusters
        plt.figure(figsize=(8, 6))
        
        # Encontrar o número ideal de clusters para os três métodos (coeficiente de silhueta)
        ideal_kmeans = range_n_clusters[np.argmax(avg_kmeans)]
        ideal_single_link = range_n_clusters[np.argmax(avg_single_link)]
        ideal_spectral = range_n_clusters[np.argmax(avg_spectral)]

        plt.plot(range_n_clusters, avg_kmeans, 'bv-', label='KMeans')
        plt.plot(range_n_clusters, avg_single_link, 'ro-', label='Single-Link')
        plt.plot(range_n_clusters, avg_spectral, 'gs-', label='Spectral Clustering')
        plt.axvline(x=ideal_kmeans, color='b', linestyle='-.', label=f'KMeans Ideal: {ideal_kmeans}')
        plt.axvline(x=ideal_single_link, color='r', linestyle='--', label=f'Single-Link Ideal: {ideal_single_link}')
        plt.axvline(x=ideal_spectral, color='g', linestyle=':', label=f'Spectral Ideal: {ideal_spectral}')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Coeficiente de Silhueta')
        plt.title(f'Coeficiente de Silhueta - Database: {database[:-5].title()}')
        plt.legend()

        plt.tight_layout()
        plt.show()



    # Aplicando PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X) 
    # Identificando as colunas que mais contribuem para cada componente principal
    pca_components = pd.DataFrame(pca.components_, columns=df.iloc[:, :-1].columns, index=['PC1', 'PC2'])
    best_column_x = pca_components.loc['PC1'].abs().nlargest(1).index.values[0]
    best_column_y = pca_components.loc['PC2'].abs().nlargest(1).index.values[0]

    def plot_clusters(ax, X, y, title):
        sc = ax.scatter(principal_components[:, 0], principal_components[:, 1], s=50, cmap='brg', c=y+1)
        ax.set_title(title)
        ax.set_xlabel(best_column_x.title())
        ax.set_ylabel(best_column_y.title())
        ax.legend(*sc.legend_elements(), title="Clusters")

    # Supondo que você tenha três conjuntos de dados
    datasets = [
        (X, y_kmeans, 'Agrupamento KMeans'), 
        (X, y_single_link, 'Agrupamento Single-Link'), 
        (X, y_spectral, 'Agrupamento Spectral Clustering')
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 linha, 3 colunas

    for ax, (X, y, title) in zip(axes, datasets):
        plot_clusters(ax, X, y, title)

    plt.tight_layout()
    plt.show()