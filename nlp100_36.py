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
ranking = sorted(word_freqency.items(),key=lambda word:word[1], reverse=True)
height = [ranking[i][1] for i in range(10)]
label = [ranking[i][0] for i in range(10)]
plt.bar(range(10), height, tick_label=label,align="center")
plt.show()
