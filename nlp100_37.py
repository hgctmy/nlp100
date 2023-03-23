import nlp100_30
import matplotlib.pyplot as plt
import japanize_matplotlib

sentences = nlp100_30.summarize_result()

neko_friends = {}
for sentence in sentences:
    for i in range(len(sentence)):
        sentence[i] = sentence[i]['base']
    if '猫' in sentence:
        for i in range(len(sentence)):
            if sentence[i] in neko_friends:
                neko_friends[sentence[i]]+=1
            elif sentence[i] != '猫':
                neko_friends[sentence[i]]= 1
ranking = sorted(neko_friends.items(),key=lambda word:word[1], reverse=True)
height = [ranking[i][1] for i in range(10)]
label = [ranking[i][0] for i in range(10)]
plt.bar(range(10), height, tick_label=label,align="center")
plt.show()
