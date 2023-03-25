import nlp100_35
import matplotlib.pyplot as plt
import japanize_matplotlib

ranking = nlp100_35.word_freqency_ranking() #単語とその頻度
plt.hist(ranking.values(), bins = 50) #単語の出現頻度のヒストグラム
plt.title("単語の出現頻度")
plt.xlabel("出現頻度")
plt.ylabel("単語の種類数")
plt.show()

#13.4身近なライブラリに親しむ,p.172