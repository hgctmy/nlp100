from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
vector = model["Spain"] - model["madrid"] + model["Athens"]  # ベクトルを計算
print(model.similar_by_vector(vector, topn=10))  # vectorと類似した上位10単語

'''% python nlp100_63.py
[('Athens', 0.6826056838035583), ('Greece', 0.4856835901737213), ('Athens_Greece', 0.46644291281700134), ('Spain', 0.4448360800743103), ('Rome', 0.4141983985900879), ('Organising_Committee_ATHOC', 0.41101545095443726), ('prosecutor_Costas_Simitzoglou', 0.40978091955184937), ('bronze_medalist_Alicia_Molik', 0.3909006714820862), ('Greek', 0.39005470275878906), ('silver_medalist_Mardy_Fish', 0.38434845209121704)]
'''
