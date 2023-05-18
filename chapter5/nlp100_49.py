from itertools import combinations
import nlp100_41

sentences = nlp100_41.load_result()
with open('ans49.txt', mode='w')as f:
    for chunks in sentences:
        nouns = []  # 名詞を含む文節
        for i, chunk in enumerate(chunks):  # 名詞を含む文節の文節番号を調べる
            if '名詞' in [morph.pos for morph in chunk.morphs]:
                nouns.append(i)
        for i, j in combinations(nouns, 2):  # 名詞節の組み合わせの経路
            path_i = []  # 構文木の根に至る経路
            path_j = []  # path_iと交わる他の枝の経路
            while i != j:
                if i < j:
                    path_i.append(i)
                    i = chunks[i].dst
                else:
                    path_j.append(j)
                    j = chunks[j].dst
            # 名詞節の組み合わせについてそれぞれの名詞をXとYに置き換え，path_iの文節を順に出力し，もしpath_jと交わっていればそれも出力
            if len(path_i) > 1:
                print('X' + ''.join([morph.surface if morph.pos != '記号' and morph.pos != '名詞' else '' for morph in chunks[path_i[0]].morphs]), end='', file=f)
                for n in path_i[1:-1]:
                    print(' -> ' + ''.join([morph.surface if morph.pos != '記号' else '' for morph in chunks[n].morphs]), end='', file=f)
                if len(path_j) > 0:
                    print(' | Y' + ''.join([morph.surface if morph.pos != '記号' and morph.pos != '名詞' else '' for morph in chunks[path_j[0]].morphs]), end='', file=f)
                    for j in path_j[1:]:
                        print(' -> ' + ''.join([morph.surface if morph.pos != '記号' else '' for morph in chunks[j].morphs]), end='', file=f)
                    print(' | ' + ''.join([morph.surface if morph.pos != '記号' else '' for morph in chunks[path_i[-1]].morphs]), end='', file=f)
                else:
                    print(' -> Y' + ''.join([morph.surface if morph.pos != '記号' and morph.pos != '名詞' else '' for morph in chunks[-1].morphs]), end='', file=f)
                print('', file=f)

# 7.7ネストを浅くする,p.93

'''
X | Yの -> 推論 -> 問題解決などの -> 知的行動を -> 代わって -> 行わせる -> 技術または | 語
X | Yや -> 推論 -> 問題解決などの -> 知的行動を -> 代わって -> 行わせる -> 技術または | 語
X | Y -> 問題解決などの -> 知的行動を -> 代わって -> 行わせる -> 技術または | 語
X | Yなどの -> 知的行動を -> 代わって -> 行わせる -> 技術または | 語
X | Yを -> 代わって -> 行わせる -> 技術または | 語
X | Yに -> 代わって -> 行わせる -> 技術または | 語
X | Yに -> 行わせる -> 技術または | 語
X | Yまたは | 語
X | Y -> コンピュータによる -> 情報処理システムの -> 実現に関する | 語
X | Yによる -> 情報処理システムの -> 実現に関する | 語
'''
