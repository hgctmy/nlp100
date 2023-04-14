import re

pattern = r'^(.*\[\[Category:.*\]\].*)$'  # "[[Category:ほにゃらら]]ほにゃらら"というパターン
with open("ukcategory.txt", mode="w")as fw, open("uk.txt", mode="r")as fr:
    data = fr.read()
    # Categoryを見つけてuk.txtに出力
    fw.write('\n'.join(re.findall(pattern, data, re.MULTILINE)))

# 6.6コードの意図を書く,p.76

'''
[[Category:イギリス|*]]
[[Category:イギリス連邦加盟国]]
[[Category:英連邦王国|*]]
[[Category:G8加盟国]]
[[Category:欧州連合加盟国|元]]
[[Category:海洋国家]]
[[Category:現存する君主国]]
[[Category:島国]]
[[Category:1801年に成立した国家・領域]]
'''
