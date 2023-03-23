import nlp100_30
import matplotlib.pyplot as plt
import math

sentences = nlp100_30.summarize_result()
word_freqency = {}
for sentence in sentences:
    for i in range(len(sentence)):
        if sentence[i]['base'] in word_freqency:
            word_freqency[sentence[i]['base']]+=1
        else:
            word_freqency[sentence[i]['base']]= 1
ranking = sorted(word_freqency.values(), reverse=True)
number = [i+1 for i in range(len(ranking))] #順位
plt.scatter(number, ranking) #散布図
plt.xscale('log') #対数に
plt.yscale('log')
plt.show()


'''
Zipfの法則とは、出現頻度がk番目に大きい要素が、
一位のものの頻度と比較して1/kに比例するという法則
'''