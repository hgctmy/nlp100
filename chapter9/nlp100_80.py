import pandas as pd
import re
import json
from collections import defaultdict


# 単語列をid列にする関数 辞書に単語が登録されていなければidは0
def id_sequence(sequence, word_id_dict):
    words = re.sub(r'[^a-zA-Z0-9]', ' ', sequence).split()
    return [word_id_dict.get(word, 0) for word in words]


if __name__ == "__main__":
    data = pd.read_table("../chapter6/train.txt")
    word_dict = defaultdict(int)  # 存在しないkeyを指定したとき新たなkeyとして追加できる
    for title in data['TITLE']:
        for word in re.sub(r'[^a-zA-Z0-9]', ' ', title).split():
            word_dict[word] += 1
    # valueで降順ソート
    word_freqency = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
    # 頻度が2以上なら順位，2未満なら0
    word_id_dict = {item[0]: i + 1 if item[1] > 1 else 0 for i, item in enumerate(word_freqency)}

    with open("word_id_dict.json", mode="w")as f:
        json.dump(word_id_dict, f)

    print(id_sequence("I am a student.", word_id_dict))

'''
[90, 5353, 17, 0]
'''
