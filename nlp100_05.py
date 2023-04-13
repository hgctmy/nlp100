str = "I am NLPer"
wordlist = str.split()
word_bigram = [[wordlist[i], wordlist[i+1]]for i in range(len(wordlist)-1)]  # 隣り合った2単語の組
charlist = list(str)
char_bigram = [[charlist[i], charlist[i+1]]for i in range(len(charlist)-1)]  # 隣り合った二文字の組

print(word_bigram)
print(char_bigram)

# 6.6コードの意図を書く,p.76

'''
% python nlp100_05.py
[['I', 'am'], ['am', 'NLPer']]
[['I', ' '], [' ', 'a'], ['a', 'm'], ['m', ' '], [' ', 'N'], ['N', 'L'], ['L', 'P'], ['P', 'e'], ['e', 'r']]
'''
