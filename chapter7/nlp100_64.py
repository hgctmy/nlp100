from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

with open("questions-words.txt", mode="r")as fr, open("questions-words.similar.txt", mode="w")as fw:
    for line in fr:
        if line[0] == ":":
            fw.write(line + '\n')
        else:  # vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)と類似度が高い単語とその類似度
            words = line.split()
            vector = model[words[1]] - model[words[0]] + model[words[2]]
            word, score = model.similar_by_vector(vector, topn=1)[0]
            # 元々の行の末に計算したベクトルと最も類似した単語とその類似度を追加したものをファイルに書き込む
            fw.write(' '.join([line, word, str(score)]) + '\n')
