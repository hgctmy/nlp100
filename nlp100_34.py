import nlp100_30

sentences = nlp100_30.summarize_result()
for sentence in sentences:
    for i in range(len(sentence)-1):
        if sentence[i]['pos'] == '名詞' and sentence[i+1]['pos'] == '名詞':
            while i < len(sentence) and sentence[i]['pos'] == '名詞':
                print(sentence[i]['surface'], end = '')
                i+=1
            print()
