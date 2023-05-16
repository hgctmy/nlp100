class Chunk:
    def __init__(self, morphs, dst):
        self.morphs = morphs
        self.dst = dst
        self.srcs = []


class Morph:
    def __init__(self, line):  # 1行を受け取ってMorphオブジェクトにする
        morph = line.split('\t')
        morph[1] = morph[1].split(',')
        self.surface = morph[0]
        self.base = morph[1][6]
        self.pos = morph[1][0]
        self.pos1 = morph[1][1]


def load_result():  # 文節番号がEOS区切りで振られているため，1文を"。"までではなく"EOS"までとした．
    sentences = []
    chunks = []  # 一文のChunkオブジェクトのリスト
    morphs = []  # 一文の係り受け解析結果のリスト
    dst = None
    with open('ai.ja.txt.parsed', mode='r') as f:
        for line in f:
            if line[0] == '*':  # ＊で始まる行なら係先インデックス番号を取得し、一つ前の文節をchunksリストに登録
                if len(morphs) > 0:
                    chunks.append(Chunk(morphs, dst))
                    morphs = []
                dst = int(line.split(' ')[2].rstrip('D'))
            elif line != 'EOS\n':  # 文末以外なら係り受け解析結果をmorphsに追加
                morphs.append(Morph(line))
            else:  # 文末なら文節をchunksリストに登録し、係もとインデックス番号を登録、1文ごとの情報をまとめる
                chunks.append(Chunk(morphs, dst))
                for i, chunk in enumerate(chunks):
                    if chunk.dst != -1:
                        chunks[chunk.dst].srcs.append(i)
                sentences.append(chunks)
                morphs = []
                chunks = []
    return sentences


if __name__ == "__main__":
    sentences = load_result()
    for chunks in sentences[:3]:
        for chunk in chunks:
            print([morph.surface for morph in chunk.morphs], chunk.dst, chunk.srcs)


# 4.7コードを段落に分割する,p.51

'''
['人工', '知能'] -1 []
[] -1 []
['人工', '知能'] 17 []
['（', 'じん', 'こうち', 'のう', '、', '、'] 17 []
['AI'] 3 []
['〈', 'エーアイ', '〉', '）', 'と', 'は', '、'] 17 [2]
['「', '『', '計算'] 5 []
['（', '）', '』', 'という'] 9 [4]
['概念', 'と'] 9 []
['『', 'コンピュータ'] 8 []
['（', '）', '』', 'という'] 9 [7]
['道具', 'を'] 10 [5, 6, 8]
['用い', 'て'] 12 [9]
['『', '知能', '』', 'を'] 12 []
['研究', 'する'] 13 [10, 11]
['計算', '機', '科学'] 14 [12]
['（', '）', 'の'] 15 [13]
['一', '分野', '」', 'を'] 16 [14]
['指す'] 17 [15]
['語', '。'] 34 [0, 1, 3, 16]
['「', '言語', 'の'] 20 []
['理解', 'や'] 20 []
['推論', '、'] 21 [18, 19]
['問題', '解決', 'など', 'の'] 22 [20]
['知的', '行動', 'を'] 24 [21]
['人間', 'に'] 24 []
['代わっ', 'て'] 26 [22, 23]
['コンピューター', 'に'] 26 []
['行わ', 'せる'] 27 [24, 25]
['技術', '」', '、', 'または', '、'] 34 [26]
['「', '計算', '機'] 29 []
['（', 'コンピュータ', '）', 'による'] 31 [28]
['知的', 'な'] 31 []
['情報処理', 'システム', 'の'] 33 [29, 30]
['設計', 'や'] 33 []
['実現', 'に関する'] 34 [31, 32]
['研究', '分野', '」', 'と', 'も'] 35 [17, 27, 33]
['さ', 'れる', '。'] -1 [34]
'''
