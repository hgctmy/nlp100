import nlp100_30

sentences = nlp100_30.summarize_result()  # 形態素解析結果
for sentence in sentences:
    for word in sentence:
        if word['pos'] == '動詞':
            print(word['surface'])

# 5.1コメントするべきではないこと,p.57

'''
長かったため最後の方だけ
し
い
感じ
得
切り落し
し
入る
死ぬ
死ん
得る
死な
得
られ
'''