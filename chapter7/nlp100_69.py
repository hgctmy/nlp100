from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.cluster import KMeans

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

with open("text.txt", mode="r")as f:
    countries = f.read().replace(" ", "_").splitlines()  # 国名リスト　スペースを"_"にし，改行を削除するためにreadからsplitlisens
countries = [country for country in countries if country in model]  # modelに含まれる国名のみ抽出
countries_vec = [model[country] for country in countries]  # 国名のベクトルリスト
cluster = KMeans(n_clusters=5).fit_predict(countries_vec)   # クラスタリング結果0~4のクラスタ番号のリスト

# t-SNEでベクトルを2次元に
tsne = TSNE(n_components=2)
countries_embedded = tsne.fit_transform(pd.DataFrame(countries_vec))

# 5つのクラスタについて色分けしたものの散布図
colors = ['red', 'blue', 'green', 'pink', 'purple']
plt.figure()
for i in range(len(countries)):
    plt.scatter(countries_embedded[i, 0], countries_embedded[i, 1], color=colors[cluster[i]])  # クラスタリングの結果から色を割り当てる
plt.show()
