from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

with open("text.txt", mode="r")as f:
    countries = f.read().replace(" ", "_").splitlines()  # 国名リスト　スペースを"_"にし，改行を削除するためにreadからsplitlisens
countries = [country for country in countries if country in model]  # modelに含まれる国名のみ抽出
countries_vec = [model[country] for country in countries]  # 国名のベクトルリスト
wardmethod = linkage(countries_vec, method='ward', metric='euclidean')  # ward法で階層型クラスタリング
plt.figure()
dendrogram(wardmethod, labels=countries)  # デンドログラムを描画
plt.show()
