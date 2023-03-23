import json

with open("uk.txt", mode="w")as f1:
    with open("jawiki-country.json", mode="r")as f2:
        for line in f2:
            line = json.loads(line)
            if line["title"] == "イギリス":
                f1.write(line["text"])
                break
