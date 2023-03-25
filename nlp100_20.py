import json

with open("uk.txt", mode="w")as fw, open("jawiki-country.json", mode="r")as fr:
    for line in fr:
        line = json.loads(line)
        if line["title"] == "イギリス":
            fw.write(line["text"])
            break

#7.7ネストを浅くする,p.93