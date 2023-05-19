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
                            source = []
                    break

# 9.3変数は一度だけ書き込む,p.123

'''
行動を代わる	に 人間に
記述をする	と 主体と
注目を集める	が サポートベクターマシンが
経験を行う	に を 元に 学習を
学習を行う	に を 元に 経験を
学習をする	て で に は を を通して なされている ACT-Rでは 元に ACT-Rでは 推論ルールを 生成規則を通して
進化を見せる	て て において は 加えて 活躍している 生成技術において 敵対的生成ネットワークは
開発を行う	は エイダ・ラブレスは
処理を行う	に に により Webに 同年に ティム・バーナーズリーにより
意味をする	に データに
処理を行う	て に 付加して コンピュータに
研究を進める	て 費やして
命令をする	で 機構で
運転をする	に 元に
特許をする	が に まで 日本が 2018年までに 2018年までに
運転をする	て に 基づいて 柔軟に
注目を集める	から は ことから ファジィは
制御を用いる	て も 受けて 他社も
制御をする	から 少なさから
改善を果たす	が で に チームが 画像処理コンテストで 2012年に
研究を続ける	が て ジェフホーキンスが 向けて
)をする	に は 8月には 8月には
注目を集める	に 急速に
投資を行う	で に 民間企業主導で 全世界的に
探索を行う	で 無報酬で
推論をする	て 経て
研究を始める	とも は マックスプランク研究所とも Googleは
研究を行う	て 始めており
開発をする	で で は 中国では 官民一体で 中国では
開発をする	で 日本で
投資をする	に は まで 2022年までに 韓国は 2022年までに
反乱を起こす	て に対して 於いて 人間に対して
監視を行う	に まで 人工知能に 歩行者まで
手続きを経る	を ウイグル族を
制御をする	は AIプログラムは
判断を介す	から 観点から
禁止を求める	が に は ヒューマン・ライツ・ウォッチが 4月には 4月には
競争を行う	は をめぐって 米国中国ロシアは 軍事利用をめぐって
追及を受ける	て で で と とともに は 暴露されており 公聴会では 整合性で 拒否すると とともに 公聴会では
研究をする	が Microsoftが
解任をする	て は 含まれており Google社員らは
解散をする	が で は 倫理委員会が 理由で Googleは
存在を見いだす	に ものに
話をする	は 哲学者は
議論を行う	まで これまで
'''
