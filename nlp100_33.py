import nlp100_30

sentences = nlp100_30.summarize_result()  # 形態素解析結果
for sentence in sentences:
    for i in range(len(sentence)-2):
        # "AのB"の形になっていれば抽出
        if (sentence[i]['pos'] == '名詞' and
            sentence[i+1]['surface'] == 'の' and
                sentence[i+2]['pos'] == '名詞'):
            print(sentence[i]['surface'] + sentence[i+1]['surface'] + sentence[i+2]['surface'])

# 4.4縦の線をまっすぐにする,p.47

'''
長かったため最後の方だけ

妻君の鼻
甕の中
烏の勘
烏の代り
吾輩の足
水の面
甕の縁
甕のふち
年の間
自然の力
水の中
座敷の上
不可思議の太平
'''
