from collections import Counter
import re
import json


with open('../corpus/brown_corpus.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# 記号を除去し、スペース + 小文字英数字 + ピリオド + カンマ だけに制限
text = text.lower()
text = text.replace('\n', ' ')
#text = re.sub(r'[^a-z\s.,]', '', text)  # ピリオドとカンマを許可
text = re.sub(r'[^a-z\s]', '', text)  
text = re.sub(r'\s+', ' ', text).strip()

# 文字bigramカウント
bigrams = Counter()
for i in range(len(text) - 1):
    bigrams[(text[i], text[i+1])] += 1

# 上位表示（確認用）
print("上位のbigram:")
for (a, b), count in bigrams.most_common(10):
    print(f"'{a}' → '{b}': {count}")

# 保存（任意）
import json
with open("char_bigrams.json", "w", encoding="utf-8") as f:
    json.dump({f"{a}{b}": count for (a, b), count in bigrams.items()}, f, indent=2)

print(f"saved to char_bigrams.json")