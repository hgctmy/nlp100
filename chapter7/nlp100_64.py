from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

with open("questions-words.txt", mode="r")as fr, open("questions-words.similar.txt", mode="w")as fw:
    for line in fr:
        if line[0] == ":":
            print(line, file=fw)
        else:
            # それぞれの単語について
            vector = model["Spain"] - model["madrid"] + model["Athens"]
            print(model.similar_by_vector(vector, topn=10), file=fw)  # vectorと類似した上位10単語
