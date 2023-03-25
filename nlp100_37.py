import nlp100_30
import matplotlib.pyplot as plt
import japanize_matplotlib

sentences = nlp100_30.summarize_result()

neko_friends = {} #猫と共起した単語とその頻度
for sentence in sentences:
    for i in range(len(sentence)):
        sentence[i] = sentence[i]['base'] #基本型を抽出
    if '猫' in sentence:
        for i in range(len(sentence)):
            #猫と共起した回数をカウント（初出の単語なら新たに登録）
            if sentence[i] in neko_friends:
                neko_friends[sentence[i]]+=1
            elif sentence[i] != '猫':
                neko_friends[sentence[i]]= 1
ranking = sorted(neko_friends.items(),key=lambda word:word[1], reverse=True) #出現頻度で降順にソート
height = [ranking[i][1] for i in range(10)]
label = [ranking[i][0] for i in range(10)]
plt.bar(range(10), height, tick_label=label,align="center") #「猫」と共起頻度の高い上位10語の棒グラフ
plt.title("「猫」と共起頻度の高い上位10語")
plt.xlabel("単語")
plt.ylabel("出現頻度")
plt.show()

#1.3小さなことは絶対にいいこと？,p.4