import nlp100_30

sentences = nlp100_30.summarize_result()
for sentence in sentences:
    for result in sentence:
        if result['pos'] == '動詞':
            print(result['base'])
