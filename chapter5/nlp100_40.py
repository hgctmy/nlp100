class Morph:
    def __init__(self, line):  # 1行を受け取ってMorphオブジェクトにする
        morph = line.split('\t')
        morph[1] = morph[1].split(',')
        self.surface = morph[0]
        self.base = morph[1][6]
        self.pos = morph[1][0]
        self.pos1 = morph[1][1]


sentences = []
morphs = []  # 一文の係り受け解析結果
with open('ai.ja.txt.parsed', mode='r') as f:
    for line in f:
        if line[0] == '*':  # ＊で始まる行をスキップ
            continue
        elif line != 'EOS\n':  # 文末でないなら係り受け解析結果をmorphsに追加
            morphs.append(Morph(line))
        else:  # 文末
            sentences.append(morphs)
            morphs = []

for sentence in sentences:
    for morph in sentence:
        print(vars(morph))

# 7.2if/elseブロックの並び順,p.86

'''
長かったため最後の方だけ
{'surface': 'は', 'base': 'は', 'pos': '助詞', 'pos1': '係助詞'}
{'surface': 'つか', 'base': 'つく', 'pos': '動詞', 'pos1': '自立'}
{'surface': 'ない', 'base': 'ない', 'pos': '助動詞', 'pos1': '*'}
{'surface': 'でしょ', 'base': 'です', 'pos': '助動詞', 'pos1': '*'}
{'surface': 'う', 'base': 'う', 'pos': '助動詞', 'pos1': '*'}
{'surface': 'ね', 'base': 'ね', 'pos': '助詞', 'pos1': '終助詞'}
{'surface': '」', 'base': '」', 'pos': '記号', 'pos1': '括弧閉'}
{'surface': 'と', 'base': 'と', 'pos': '助詞', 'pos1': '格助詞'}
{'surface': '答え', 'base': '答える', 'pos': '動詞', 'pos1': '自立'}
{'surface': 'て', 'base': 'て', 'pos': '助詞', 'pos1': '接続助詞'}
{'surface': 'いる', 'base': 'いる', 'pos': '動詞', 'pos1': '非自立'}
{'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}
'''
