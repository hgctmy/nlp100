def summarize_result():
    results = []
    sentences = []
    with open("neko.txt.mecab", mode = "r") as f:
        for line in f:
            if line != "EOS\n":
                wordinfo = line.split("\t")
                if wordinfo[0]!='\n' and wordinfo[0]!='':
                    partofspeech = wordinfo[1].split(",")
                    result = {"surface":wordinfo[0], "base":partofspeech[6], "pos":partofspeech[0], "pos1":partofspeech[1]}
                    results.append(result)
            else:
                sentences.append(results)
                results = []
    return sentences

if __name__ == "__main__":
    answer = summarize_result()
    print(answer)
