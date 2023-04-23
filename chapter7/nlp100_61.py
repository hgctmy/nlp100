from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
print(model.similarity('United_States', 'U.S.'))  # コサイン類似度

'''
% python nlp100_61.py
0.7310774
'''
