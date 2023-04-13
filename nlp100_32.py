import nlp100_30

sentences = nlp100_30.summarize_result()  # 形態素解析結果
for sentence in sentences:
    for word in sentence:
        if word['pos'] == '動詞':
            print(word['base'])

# 5.1コメントするべきではないこと,p.57

'''
長かったため最後の方だけ

する
なる
くる
つく
いる
いる
する
いる
感じる
得る
切り落す
する
入る
死ぬ
死ぬ
得る
死ぬ
得る
られる
'''
