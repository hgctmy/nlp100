import nlp100_30

sentences = nlp100_30.summarize_result()
for sentence in sentences:
    for i in range(len(sentence)-2):
        if sentence[i]['pos'] == '名詞' and sentence[i+1]['surface']=='の' and sentence[i+2]['pos']=='名詞':
            print(sentence[i]['surface'] + sentence[i+1]['surface'] + sentence[i+2]['surface'])

