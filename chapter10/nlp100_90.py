import json
from collections import defaultdict


# 単語列をid列にする関数 辞書に単語が登録されていなければidは0
def id_sequence(sequence, word_id_dict):
    words = sequence.split()
    return [word_id_dict.get(word, 0) for word in words]


if __name__ == "__main__":
    # 英語
    with open("kftt-data-1.0/data/tok/kyoto-train.cln.en")as f:
        data = [line.rstrip() for line in f.readlines()]
    word_dict_en = defaultdict(int)  # 存在しないkeyを指定したとき新たなkeyとして追加できる
    for sequence in data:
        for word in sequence.split():
            word_dict_en[word] += 1
    # valueで降順ソート
    word_freqency = sorted(word_dict_en.items(), key=lambda x: x[1], reverse=True)
    # 頻度が2以上なら順位，2未満なら0
    word_id_dict_en = {item[0]: i + 1 for i, item in enumerate(word_freqency)}

    with open("word_id_dict_en.json", mode="w")as f:
        json.dump(word_id_dict_en, f)

    # 日本語
    with open("kftt-data-1.0/data/tok/kyoto-train.cln.ja")as f:
        data = [line.rstrip() for line in f.readlines()]
    word_dict_ja = defaultdict(int)  # 存在しないkeyを指定したとき新たなkeyとして追加できる
    for sequence in data:
        for word in sequence.split():
            word_dict_ja[word] += 1
    # valueで降順ソート
    word_freqency = sorted(word_dict_ja.items(), key=lambda x: x[1], reverse=True)
    # 頻度が2以上なら順位，2未満なら0
    word_id_dict_ja = {item[0]: i + 1 for i, item in enumerate(word_freqency)}

    with open("word_id_dict_ja.json", mode="w")as f:
        json.dump(word_id_dict_ja, f, ensure_ascii=False)

    print(id_sequence("I am a student .", word_id_dict_en))
    print(id_sequence("私 は 学生 だ 。", word_id_dict_ja))

'''
[314, 3304, 11, 1554, 3]
[1363, 5, 1309, 79, 2]
'''
