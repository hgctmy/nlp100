import nlp100_35
import matplotlib.pyplot as plt
import math
import japanize_matplotlib

ranking = nlp100_35.word_freqency_ranking() #単語とその頻度
number = [i+1 for i in range(len(ranking))] #順位
plt.scatter(number, ranking.values()) #散布図
plt.xscale('log') #対数に
plt.yscale('log')
plt.title("頻度の順位と頻度の両対数グラフ")
plt.xlabel("出現頻度順位")
plt.ylabel("出現頻度")
plt.show()

#13.4身近なライブラリに親しむ,p.172
'''
Zipfの法則とは、出現頻度がk番目に大きい要素が、
一位のものの頻度と比較して1/kに比例するという法則
'''