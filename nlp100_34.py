import nlp100_30

sentences = nlp100_30.summarize_result()  # 形態素解析結果
for sentence in sentences:
    for i in range(len(sentence)-1):
        # 名詞の連節を最長一致で抽出
        if sentence[i]['pos'] == '名詞' and sentence[i+1]['pos'] == '名詞':
            # 名詞かどうかの評価が重複してしまっている
            while i < len(sentence) and sentence[i]['pos'] == '名詞':
                print(sentence[i]['surface'], end='')
                i += 1
            print()

# 5.2自分の考えを記録する,p.60
