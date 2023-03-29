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
