import nlp100_41

sentences = nlp100_41.load_result()
for chunks in sentences:
    for chunk in chunks:
        if chunk.dst != -1:
            source = ''.join([morph.surface if morph.pos != '記号' else '' for morph in chunk.morphs])  # 係元
            destination = ''.join(
                [morph.surface if morph.pos != '記号' else '' for morph in chunks[chunk.dst].morphs])  # 係先
            print(source + '\t' + destination)

# 7.3三項演算子,p.88

'''
長かったため最後の方だけ

おかげで        明らかになったとは
意見の  違いは
違いは  明らかになったとは
明らかになったとは      思いますが
思いますが      つくのでしょうかと
果たして        つくのでしょうかと
何か    つくのでしょうかと
決着が  つくのでしょうかと
つくのでしょうかと      発言し
発言し  答えている
伊勢田は        答えている
決着は  つかないでしょうねと
つかないでしょうねと    答えている
'''
