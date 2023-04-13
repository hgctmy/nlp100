import nlp100_41

sentences = nlp100_41.load_result()
with open('ans45.txt', mode='w')as f:
    for chunks in sentences:
        for chunk in chunks:
            for morph in chunk.morphs:
                source = []  # 係先
                if morph.pos == '動詞':
                    verb = morph.base  # 動詞を含む文節の最左の動詞
                    for src in chunk.srcs:  # 係元に助詞があれば
                        for morph in chunks[src].morphs:
                            if morph.pos == '助詞':
                                source.append(morph.surface)
                    if len(source) > 0:  # 動詞とそれに係る助詞（格）を出力
                        print(verb + '\t' + ' '.join(sorted(list(set(source)))), file=f)
                    break

# 6.6コードの意図を書く,p.76

'''
用いる	を
する	て を
指す	を
代わる	に を
行う	て に
する	と も
述べる	で に の は
する	で を
する	を
する	を
'''
