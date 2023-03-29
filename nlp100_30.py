def summarize_result():
    sentence = []  # 1文の情報
    sentences = []  # 文章全体
    with open("neko.txt.mecab", mode="r") as f:
        for line in f:
            if line != "EOS\n":  # 文の最後でないなら
                wordinfo = line.split("\t")  # タブ区切りで分割
                if wordinfo[0] != '\n' and wordinfo[0] != '':
                    partofspeech = wordinfo[1].split(",")  # 品詞情報
                    result = {"surface": wordinfo[0], "base": partofspeech[6], "pos": partofspeech[0], "pos1": partofspeech[1]}  # 1つの単語の形態素解析結果
                    sentence.append(result)
            else:  # 文が終わったなら
                sentences.append(sentence)
                sentence = []  # 初期化
    return sentences


if __name__ == "__main__":
    answer = summarize_result()
    print(answer)

# 10.5プロジェクトに特化した機能,p.136
