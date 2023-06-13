import json
from collections import defaultdict


# 単語列をid列にする関数 辞書に単語が登録されていなければidは0
def sequence2id(sequence, word_id_dict):
    words = sequence.split()
    return [word_id_dict.get(word, 0) for word in words]


def id2seqense(ids, word_id_dict):
    return ' '.join([[k for k, v in word_id_dict.items() if v == id][0] for id in ids])


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
    word_id_dict_en = {'<unk>': 0, '<pad>': 1, '<start>': 2, '<end>': 3}
    for i, item in enumerate(word_freqency):
        word_id_dict_en[item[0]] = i + 4
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
    word_id_dict_ja = {'<unk>': 0, '<pad>': 1, '<start>': 2, '<end>': 3}
    for i, item in enumerate(word_freqency):
        word_id_dict_ja[item[0]] = i + 4
    with open("word_id_dict_ja.json", mode="w")as f:
        json.dump(word_id_dict_ja, f, ensure_ascii=False)

    print(sequence2id("I am a student .", word_id_dict_en))
    print(sequence2id("私 は 学生 だ 。", word_id_dict_ja))
    print(id2seqense([317, 3307, 14, 1557, 6], word_id_dict_en))

'''
[317, 3307, 14, 1557, 6]
[1366, 8, 1312, 82, 5]
I am a student .
'''
