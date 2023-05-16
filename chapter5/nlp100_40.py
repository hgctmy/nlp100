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
        if line[0] == '*' or line == 'EOS\n':  # ＊で始まる行をスキップ
            continue
        elif line[0] != '。':  # 文末でないなら係り受け解析結果をmorphsに追加
            morphs.append(Morph(line))
        else:  # 文末
            morphs.append(Morph(line))  # 。を追加
            sentences.append(morphs)
            morphs = []

for sentence in sentences[:1]:
    for morph in sentence:
        print(vars(morph))

# 7.2if/elseブロックの並び順,p.86

'''
% python nlp100_40.py
{'surface': '人工', 'base': '人工', 'pos': '名詞', 'pos1': '一般'}
{'surface': '知能', 'base': '知能', 'pos': '名詞', 'pos1': '一般'}
{'surface': '人工', 'base': '人工', 'pos': '名詞', 'pos1': '一般'}
{'surface': '知能', 'base': '知能', 'pos': '名詞', 'pos1': '一般'}
{'surface': '（', 'base': '（', 'pos': '記号', 'pos1': '括弧開'}
{'surface': 'じん', 'base': 'じん', 'pos': '名詞', 'pos1': '一般'}
{'surface': 'こうち', 'base': 'こうち', 'pos': '名詞', 'pos1': '一般'}
{'surface': 'のう', 'base': 'のう', 'pos': '助詞', 'pos1': '終助詞'}
{'surface': '、', 'base': '、', 'pos': '記号', 'pos1': '読点'}
{'surface': '、', 'base': '、', 'pos': '記号', 'pos1': '読点'}
{'surface': 'AI', 'base': '*\n', 'pos': '名詞', 'pos1': '一般'}
{'surface': '〈', 'base': '〈', 'pos': '記号', 'pos1': '括弧開'}
{'surface': 'エーアイ', 'base': '*\n', 'pos': '名詞', 'pos1': '固有名詞'}
{'surface': '〉', 'base': '〉', 'pos': '記号', 'pos1': '括弧閉'}
{'surface': '）', 'base': '）', 'pos': '記号', 'pos1': '括弧閉'}
{'surface': 'と', 'base': 'と', 'pos': '助詞', 'pos1': '格助詞'}
{'surface': 'は', 'base': 'は', 'pos': '助詞', 'pos1': '係助詞'}
{'surface': '、', 'base': '、', 'pos': '記号', 'pos1': '読点'}
{'surface': '「', 'base': '「', 'pos': '記号', 'pos1': '括弧開'}
{'surface': '『', 'base': '『', 'pos': '記号', 'pos1': '括弧開'}
{'surface': '計算', 'base': '計算', 'pos': '名詞', 'pos1': 'サ変接続'}
{'surface': '（', 'base': '（', 'pos': '記号', 'pos1': '括弧開'}
{'surface': '）', 'base': '）', 'pos': '記号', 'pos1': '括弧閉'}
{'surface': '』', 'base': '』', 'pos': '記号', 'pos1': '括弧閉'}
{'surface': 'という', 'base': 'という', 'pos': '助詞', 'pos1': '格助詞'}
{'surface': '概念', 'base': '概念', 'pos': '名詞', 'pos1': '一般'}
{'surface': 'と', 'base': 'と', 'pos': '助詞', 'pos1': '並立助詞'}
{'surface': '『', 'base': '『', 'pos': '記号', 'pos1': '括弧開'}
{'surface': 'コンピュータ', 'base': 'コンピュータ', 'pos': '名詞', 'pos1': '一般'}
{'surface': '（', 'base': '（', 'pos': '記号', 'pos1': '括弧開'}
{'surface': '）', 'base': '）', 'pos': '記号', 'pos1': '括弧閉'}
{'surface': '』', 'base': '』', 'pos': '記号', 'pos1': '括弧閉'}
{'surface': 'という', 'base': 'という', 'pos': '助詞', 'pos1': '格助詞'}
{'surface': '道具', 'base': '道具', 'pos': '名詞', 'pos1': '一般'}
{'surface': 'を', 'base': 'を', 'pos': '助詞', 'pos1': '格助詞'}
{'surface': '用い', 'base': '用いる', 'pos': '動詞', 'pos1': '自立'}
{'surface': 'て', 'base': 'て', 'pos': '助詞', 'pos1': '接続助詞'}
{'surface': '『', 'base': '『', 'pos': '記号', 'pos1': '括弧開'}
{'surface': '知能', 'base': '知能', 'pos': '名詞', 'pos1': '一般'}
{'surface': '』', 'base': '』', 'pos': '記号', 'pos1': '括弧閉'}
{'surface': 'を', 'base': 'を', 'pos': '助詞', 'pos1': '格助詞'}
{'surface': '研究', 'base': '研究', 'pos': '名詞', 'pos1': 'サ変接続'}
{'surface': 'する', 'base': 'する', 'pos': '動詞', 'pos1': '自立'}
{'surface': '計算', 'base': '計算', 'pos': '名詞', 'pos1': 'サ変接続'}
{'surface': '機', 'base': '機', 'pos': '名詞', 'pos1': '接尾'}
{'surface': '科学', 'base': '科学', 'pos': '名詞', 'pos1': '一般'}
{'surface': '（', 'base': '（', 'pos': '記号', 'pos1': '括弧開'}
{'surface': '）', 'base': '）', 'pos': '記号', 'pos1': '括弧閉'}
{'surface': 'の', 'base': 'の', 'pos': '助詞', 'pos1': '連体化'}
{'surface': '一', 'base': '一', 'pos': '名詞', 'pos1': '数'}
{'surface': '分野', 'base': '分野', 'pos': '名詞', 'pos1': '一般'}
{'surface': '」', 'base': '」', 'pos': '記号', 'pos1': '括弧閉'}
{'surface': 'を', 'base': 'を', 'pos': '助詞', 'pos1': '格助詞'}
{'surface': '指す', 'base': '指す', 'pos': '動詞', 'pos1': '自立'}
{'surface': '語', 'base': '語', 'pos': '名詞', 'pos1': '一般'}
{'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}
'''
