import pickle

with open('model.pickle', mode='rb') as f:
    lr = pickle.load(f)  # 学習したモデルを読み込む

coef = lr.coef_
with open("train.feature.txt", mode="r") as f:
    word = f.readline().rstrip().split("\t")  # 特徴量のラベルを取得

for i in range(len(lr.classes_)):
    coef_sorted = list(zip(*sorted(zip(coef[i], word), reverse=True)))[1]  # 特徴量について，重みが大きい順にソートしたもののラベル
    print("\nカテゴリ: ", lr.classes_[i], "\n上位10", coef_sorted[:10], "\n下位10", coef_sorted[:-10:-1])

'''
% python nlp100_57.py

カテゴリ:  b
上位10 ('china', 'update', 'fed', 'bank', 'stocks', 'ecb', 'euro', 'us', 'as', 'ukraine')
下位10 ('the', 'and', 'her', 'google', 'ebola', 'apple', 'kardashian', 'video', 'she')

カテゴリ:  e
上位10 ('kardashian', 'her', 'she', 'and', 'chris', 'star', 'kim', 'the', 'miley', 'cyrus')
下位10 ('update', 'us', 'google', 'says', 'china', 'gm', 'facebook', 'ceo', 'apple')

カテゴリ:  m
上位10 ('ebola', 'study', 'cancer', 'drug', 'fda', 'mers', 'could', 'health', 'virus', 'cases')
下位10 ('gm', 'google', 'apple', 'facebook', 'at', 'ceo', 'deal', 'china', 'kardashian')

カテゴリ:  t
上位10 ('google', 'apple', 'facebook', 'climate', 'gm', 'microsoft', 'mobile', 'tesla', 'nasa', 'comcast')
下位10 ('stocks', 'fed', 'her', 'kardashian', 'drug', 'cancer', 'ecb', 'bank', 'ebola')
'''
