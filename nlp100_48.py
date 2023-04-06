import nlp100_41

sentences = nlp100_41.load_result()
with open('ans48.txt', mode='w')as f:
    for chunks in sentences:
        for chunk in chunks:
            for morph in chunk.morphs:  # 名詞を含む文節を見つけ、出力する
                if morph.pos == '名詞':
                    noun = ''.join([morph.surface if morph.pos != '記号' else '' for morph in chunk.morphs])
                    print(noun, end='', file=f)
                    destination = chunk.dst
                    while destination != -1:  # 係先が無くなるまで係先を出力
                        print(' -> ' + ''.join([morph.surface if morph.pos != '記号' else '' for morph in chunks[destination].morphs]), end='', file=f)
                        destination = chunks[destination].dst
                    print('', file=f)
                    break

# 6.6コードの意図を書く,p.76
