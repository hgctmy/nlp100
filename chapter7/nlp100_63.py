from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
vector = model["Spain"] - model["madrid"] + model["Athens"]  # ベクトルを計算
print(model.similar_by_vector(vector, topn=10))  # vectorと類似した上位10単語
