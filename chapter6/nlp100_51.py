from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def create_features(kinds):
    data = pd.read_table(f'{kinds}.txt')
    X = data["TITLE"].map(lambda x: x.lower())  # 小文字化したタイトル
    # 単語ユニグラム，バイグラムについてtf-idfでベクトル化
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vector = vectorizer.fit_transform(X)  # ベクトル化したもの
    feature = pd.DataFrame(vector.toarray(), columns=vectorizer.get_feature_names_out())
    feature.to_csv(f'{kinds}.feature.txt', sep="\t", index=False)


create_features(kinds='train')
create_features(kinds='valid')
create_features(kinds='test')
