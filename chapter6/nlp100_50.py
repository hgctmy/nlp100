import pandas as pd

df = pd.read_table("NewsAggregatorDataset/newsCorpora.csv")

# ”Reuters”, “Huffington Post”, “Businessweek”, “Contactmusic.com”, “Daily Mail”の事例（記事）のカテゴリとタイトルを抽出
data = df[df.iloc[:, 3].isin(["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"])].iloc[:, [4, 1]]

shuffled = data.sample(frac=1)  # ランダムに並べ替える
# 8割，1割，1割で訓練データ，検証データ，テストデータに分割する
train = shuffled[:round(shuffled.shape[0]*0.8)]
valid = shuffled[round(shuffled.shape[0]*0.8):round(df.shape[0]*0.9)]
test = shuffled[round(shuffled.shape[0]*0.9):]

train.to_csv("train.txt", sep="\t", header=["CATEGORY", "TITLE"], index=False)
valid.to_csv("valid.txt", sep="\t", header=["CATEGORY", "TITLE"], index=False)
test.to_csv("test.txt", sep="\t", header=["CATEGORY", "TITLE"], index=False)
