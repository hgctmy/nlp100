import random

def shuffle_sentence(sentence):
    words = sentence.split()
    answer = []
    for word in words:
        if len(word)>4: #長さが4以上なら先頭と末尾以外をシャッフル
            word = word[:1] + "".join(random.sample(word[1:-1], len(word)-2)) + word[-1:]
        answer.append(word)
    print(" ".join(answer)) #文に

str = "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
shuffle_sentence(str)

#2.1明確な単語を選ぶ,p.10