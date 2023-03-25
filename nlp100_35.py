import nlp100_30

def word_freqency_ranking(Reverse = 1): #Reverseが1なら降順、0なら昇順
    sentences = nlp100_30.summarize_result() #形態素解析結果
    word_freqency = {} #単語とその頻度
    for sentence in sentences:
        for i in range(len(sentence)):
            if sentence[i]['base'] in word_freqency: #すでに出てきた単語ならカウントを増やす
                word_freqency[sentence[i]['base']]+=1
            else: #初出の単語なら単語を登録
                word_freqency[sentence[i]['base']]= 1
    ranking = sorted(word_freqency.items(),key=lambda word:word[1], reverse = Reverse ) #頻度の多い順にソート
    ranking_dict = {ranking[i][0]: ranking[i][1] for i in range(len(ranking))} #辞書型に
    return ranking_dict

if __name__ == "__main__":
    answer = word_freqency_ranking()
    print(answer)

#13.4身近なライブラリに親しむ,p.172