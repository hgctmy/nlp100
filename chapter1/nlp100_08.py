def cipher(str):
    rep = [chr(219-ord(x)) if x.islower() else x for x in str]  # 小文字なら変換
    return ''.join(rep)  # リスト->文字列


message = "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
message = cipher(message)
print("暗号化:", message)
message = cipher(message)
print("複号化:", message)

# 7.3三項演算子,p.88

'''
% python nlp100_08.py
暗号化: gsv jfrxp yildm ulc qfnkh levi gsv ozab wlt
複号化: the quick brown fox jumps over the lazy dog
'''