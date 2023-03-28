import nlp100_41

sentences = nlp100_41.load_result()
for chunks in sentences:
    for chunk in chunks:
        if chunk.dst != -1:
            source = ''.join([morph.surface if morph.pos != '記号' else '' for morph in chunk.morphs]) #係元
            destination = ''.join([morph.surface if morph.pos != '記号' else '' for morph in chunks[chunk.dst].morphs]) #係先
            print(source + '\t' + destination)

#7.3三項演算子,p.88