import nlp100_35
import matplotlib.pyplot as plt
import japanize_matplotlib

ranking = nlp100_35.word_freqency_ranking() #単語とその頻度（降順）
height = list(ranking.values())[:10]
label = list(ranking.keys())[:10]
plt.bar(range(10), height, tick_label=label,align="center") #出現頻度上位10語の棒グラフ
plt.title("単語の出現頻度上位10")
plt.xlabel("単語")
plt.ylabel("出現頻度")
plt.show()

#10.5プロジェクトに特化した機能,p.136