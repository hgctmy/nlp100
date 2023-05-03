import pandas as pd
from scipy.stats import spearmanr
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

df = pd.read_csv("wordsim353/combined.csv")
df['vector_sim'] = [model.similarity(df.iloc[i, 0], df.iloc[i, 1])for i in range(df.shape[0])]  # ２つの単語のベクトルのコサイン類似度
df = df.rank(numeric_only=True, ascending=False)  # 数値を降順での順位に
print(spearmanr(df['Human (mean)'], df['vector_sim']))

'''
% python nlp100_66.py
SpearmanrResult(correlation=0.7000166486272194, pvalue=2.86866666051422e-53)
'''
