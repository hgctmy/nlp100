import nlp100_30
import matplotlib.pyplot as plt
import japanize_matplotlib

sentences = nlp100_30.summarize_result()
word_freqency = {}
for sentence in sentences:
    for i in range(len(sentence)):
        if sentence[i]['base'] in word_freqency:
            word_freqency[sentence[i]['base']]+=1
        else:
            word_freqency[sentence[i]['base']]= 1
ranking = sorted(word_freqency.values())
plt.hist(ranking, bins = 50)
plt.show()
