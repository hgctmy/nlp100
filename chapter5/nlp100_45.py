import nlp100_41

sentences = nlp100_41.load_result()
with open('ans45.txt', mode='w')as f:
    for chunks in sentences:
        for chunk in chunks:
            for morph in chunk.morphs:
                source = []  # 係元
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
 sort ans45.txt | uniq -c | sort -n -r | head
  49 する       を
  19 する       が
  15 する       に
  15 する       と
  12 する       は を
  10 する       に を
   9 する       で を
   9 よる       に
   8 する       が に
   8 行う       を

% grep 行う ans45.txt | uniq -c | sort -n -r
   4 行う       を
   1 行う       まで を
   1 行う       から
   1 行う       に により を
   1 行う       に まで を
   1 行う       は を をめぐって
   1 行う       が て で に は
   1 行う       が で に は
   1 行う       で に を
   1 行う       て に を
   1 行う       て に は
   1 行う       が で は
   1 行う       は を
   1 行う       に を
   1 行う       で を
   1 行う       て を
   1 行う       て に
   1 行う       を
   1 行う       を
   1 行う       を
   1 行う       を
   1 行う       に


grep なる ans45.txt | uniq -c | sort -n -r
   2 なる       に は
   2 なる       と
   1 無くなる   は
   1 異なる     が で
   1 異なる     も
   1 なる       から が て で と は
   1 なる       から で と
   1 なる       て として に は
   1 なる       が と にとって は
   1 なる       で と など は
   1 なる       が で と に は
   1 なる       に は も
   1 なる       で に は
   1 なる       て に は
   1 なる       が に は
   1 なる       が て と
   1 なる       に は
   1 なる       で は
   1 なる       が に
   1 なる       が と
   1 なる       が と
   1 なる       が と
   1 なる       も
   1 なる       は
   1 なる       に
   1 なる       に


grep 与える ans45.txt | uniq -c | sort -n -r
   1 与える     が など に
   1 与える     に は を
   1 与える     が に
'''
