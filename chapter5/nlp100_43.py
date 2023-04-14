import nlp100_41

sentences = nlp100_41.load_result()
for chunks in sentences:
    for chunk in chunks:
        if chunk.dst != -1:
            source = ''.join([morph.surface if morph.pos != '記号' else '' for morph in chunk.morphs])  # 係元
            source_pos = [morph.pos for morph in chunk.morphs]  # 係元の品詞
            destination = ''.join([morph.surface if morph.pos != '記号' else '' for morph in chunks[chunk.dst].morphs])  # 係先
            destination_pos = [morph.pos for morph in chunks[chunk.dst].morphs]  # 係先の品詞
            if '名詞' in source_pos and '動詞' in destination_pos:  # 係元に動詞が含まれ、係先に名詞が含まれるなら
                print(source + '\t' + destination)

# 2.1明確な単語を選ぶ,p.10

'''
長かったため最後の方だけ

須藤は  発言し
これまで        行ってきました
長時間  行ってきました
議論を  行ってきました
おかげで        明らかになったとは
違いは  明らかになったとは
明らかになったとは      思いますが
何か    つくのでしょうかと
決着が  つくのでしょうかと
つくのでしょうかと      発言し
発言し  答えている
伊勢田は        答えている
決着は  つかないでしょうねと
'''
