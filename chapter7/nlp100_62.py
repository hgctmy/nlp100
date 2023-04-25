from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
print(model.most_similar('United_States', topn=10))  # コサイン類似度

'''
[('Unites_States', 0.7877248525619507), ('Untied_States', 0.7541369199752808), ('United_Sates', 0.7400726079940796), ('U.S.', 0.7310774326324463), ('theUnited_States', 0.6404393911361694), ('America', 0.6178410053253174), ('UnitedStates', 0.6167311668395996), ('Europe', 0.613298773765564), ('countries', 0.604480504989624), ('Canada', 0.6019070148468018)]
'''
