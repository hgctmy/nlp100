def cipher(str):
    rep = [chr(219-ord(x)) if x.islower() else x for x in str]  # 小文字なら変換
    return ''.join(rep)  # リスト->文字列


message = "the quick brown fox jumps over the lazy dog"
message = cipher(message)
print("暗号化:", message)
message = cipher(message)
print("複号化:", message)

# 7.3三項演算子,p.88
