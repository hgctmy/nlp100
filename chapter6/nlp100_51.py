from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# trainデータについて
data = pd.read_table('train.txt')
X = data["TITLE"].map(lambda x: x.lower())  # 小文字化したタイトル
# 単語ユニグラム，バイグラムについてtf-idfでベクトル化
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
vector = vectorizer.fit_transform(X)  # ベクトル化したもの
feature = pd.DataFrame(vector.toarray(), columns=vectorizer.get_feature_names_out())
feature.to_csv('train.feature.txt', sep="\t", index=False)

# validデータについて
data = pd.read_table('valid.txt')
X = data["TITLE"].map(lambda x: x.lower())  # 小文字化したタイトル
vector = vectorizer.transform(X)  # ベクトル化したもの
feature = pd.DataFrame(vector.toarray(), columns=vectorizer.get_feature_names_out())
feature.to_csv('valid.feature.txt', sep="\t", index=False)

# testデータについて
data = pd.read_table('test.txt')
X = data["TITLE"].map(lambda x: x.lower())  # 小文字化したタイトル
vector = vectorizer.transform(X)  # ベクトル化したもの
feature = pd.DataFrame(vector.toarray(), columns=vectorizer.get_feature_names_out())
feature.to_csv('test.feature.txt', sep="\t", index=False)

'''
0.0	0.0	0.0	0.0	0.0	0.0	0.0 ...
'''
