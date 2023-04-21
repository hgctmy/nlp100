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

'''
CATEGORY	TITLE
b	UPDATE 1-BG sells UK CATS gas pipeline stake to infrastructure fund
e	Justin Bieber and Selena Gomez continue to rekindle their on-off relationship as  ...
b	Allergan Investors Left Wanting More After Valeant Bid: Real M&A
b	WRAPUP 2-New home sales fall, but US economy stays on solid ground
e	'Gone Girl' Trailer: The Meaning Of Ben Affleck's Life Is She
m	Johnson & Johnson Pulls Controversial Device That May Spread Cancer
b	FOREX-Euro back around $1.37 as ECB meets again
e	Teenage girl, 17, was 'raped at Keith Urban concert in front of hundreds of fans  ...
e	Bryan Singer - Bryan Singer's Lawyer Dismisses Lawsuit As Absurd

'''
