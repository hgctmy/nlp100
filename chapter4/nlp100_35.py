import nlp100_30


def word_freqency_ranking(Reverse=1):  # Reverseが1なら降順、0なら昇順
    sentences = nlp100_30.summarize_result()  # 形態素解析結果
    word_freqency = {}  # 単語とその頻度
    for sentence in sentences:
        for i in range(len(sentence)):
            if sentence[i]['base'] in word_freqency:  # すでに出てきた単語ならカウントを増やす
                word_freqency[sentence[i]['base']] += 1
            else:  # 初出の単語なら単語を登録
                word_freqency[sentence[i]['base']] = 1
    ranking = sorted(word_freqency.items(), key=lambda word: word[1], reverse=Reverse)  # 頻度の多い順にソート
    ranking_dict = {item[0]: item[1] for item in ranking}  # 辞書型に
    return ranking_dict


if __name__ == "__main__":
    print(word_freqency_ranking())

# 13.4身近なライブラリに親しむ,p.172

'''
長かったため最後の方だけ
冥土': 1, '対面': 1, 'しるし': 1, '不孝': 1, '有': 1, '郷': 1, '帰臥': 1, 'がたつく': 1, '月夜': 1, '茶色': 1, '月影': 1, '液体': 1, '熱苦しい': 1, '息遣い': 1, '墓場': 1, '悔やむ': 1, '口癖': 1, '良薬': 1, 'にがい': 1, '空前': 1, 'ぴりぴり': 1, '難なく': 1, 'ぽうっと': 1, '引掻く': 1, 'よたよた': 1, '水の上': 1, '掻ける': 1, '減': 1, '拷問': 1}
'''
