import nlp100_30

sentences = nlp100_30.summarize_result()
word_freqency = {}
for sentence in sentences:
    for i in range(len(sentence)):
        if sentence[i]['base'] in word_freqency:
            word_freqency[sentence[i]['base']]+=1
        else:
            word_freqency[sentence[i]['base']]= 1
word_freqency = sorted(word_freqency.items(),key=lambda word:word[1], reverse=True)
print(word_freqency)
