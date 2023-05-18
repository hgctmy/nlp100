import nlp100_41

sentences = nlp100_41.load_result()
with open('ans47.txt', mode='w')as f:
    for chunks in sentences:
        for chunk in chunks:
            for morph in chunk.morphs:
                if morph.pos == '動詞':  # 動詞を見つける
                    for i, src in enumerate(chunk.srcs):  # サ変接続+を かどうかを調べる
                        for j in range(len(chunks[src].morphs) - 1):
                            source = []  # 係元
                            paragraph = []  # 項
                            if chunks[src].morphs[j].pos1 == 'サ変接続' and chunks[src].morphs[j + 1].surface == 'を':
                                # 最左のサ変接続名詞+を+動詞
                                functionverb = chunks[src].morphs[j].surface + chunks[src].morphs[j + 1].surface + morph.base
                                # 同じ動詞に係るサ変接続名詞+を の形になっていない助詞を探し、格と項を取得
                                for src2 in chunk.srcs[:i] + chunk.srcs[i + 1:]:
                                    for morph2 in chunks[src2].morphs:
                                        if morph2.pos == '助詞':
                                            source.append(morph2.surface)
                                            paragraph.append(''.join([morph3.surface for morph3 in chunks[src2].morphs if morph3.pos != '記号']))
                    if len(source) > 0:  # サ変接続+を+動詞とそれに係る格と項を出力
                        source, paragraph = zip(*sorted(zip(source, paragraph)))  # 辞書順にソート
                        print(functionverb + '\t' + ' '.join(source) + ' ' + ' '.join(paragraph), file=f)
                    break

# 9.3変数は一度だけ書き込む,p.123

'''
注目を集める	が サポートベクターマシンが
学習を行う	に を 元に 経験を
進化を見せる	て て において は 加えて 活躍している 生成技術において 敵対的生成ネットワークは
開発を行う	は エイダ・ラブレスは
処理を行う	に に により Webに 同年に ティム・バーナーズリーにより
意味をする	に データに
処理を行う	て に 付加して コンピュータに
研究を進める	て 費やして
研究を進める	て 費やして
運転をする	に 元に
'''
