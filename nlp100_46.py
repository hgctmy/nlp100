import nlp100_41

sentences = nlp100_41.load_result()
with open('ans46.txt', mode='w')as f:
    for chunks in sentences:
        for chunk in chunks:
            for morph in chunk.morphs:
                source = []  # 係先
                paragraph = []
                if morph.pos == '動詞':
                    verb = morph.base  # 動詞を含む文節の最左の動詞
                    for src in chunk.srcs:  # 係元に助詞があれば格と項を取得
                        for morph in chunks[src].morphs:
                            if morph.pos == '助詞':
                                source.append(morph.surface)
                                paragraph.append(''.join([morph.surface if morph.pos != '記号' else '' for morph in chunks[src].morphs]))
                    if len(source) > 0:  # 動詞とそれに係る格と項を出力
                        source, paragraph = zip(*sorted(zip(source, paragraph)))  # 辞書順にソート
                        print(verb + '\t' + ' '.join(source) + ' ' + ' '.join(paragraph), file=f)
                    break

# 6.6コードの意図を書く,p.76
