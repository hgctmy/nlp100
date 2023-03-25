import nlp100_30

sentences = nlp100_30.summarize_result() #形態素解析結果
for sentence in sentences:
    for i in range(len(sentence)-2):
        #"AのB"の形になっていれば抽出
        if (sentence[i]['pos']      == '名詞' and 
            sentence[i+1]['surface']== 'の' and 
            sentence[i+2]['pos']    == '名詞'):
            print(sentence[i]['surface'] + sentence[i+1]['surface'] + sentence[i+2]['surface'])

    #4.4縦の線をまっすぐにする,p.47