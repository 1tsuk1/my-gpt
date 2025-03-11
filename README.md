```
import re

data = ["24ヶ月", "60ヶ月", "12ヶ月"]

# 数値を抽出してリスト化し、ソート
sorted_data = sorted(data, key=lambda s: int(re.search(r'\d+', s).group()))

print(sorted_data)  # ['12ヶ月', '24ヶ月', '60ヶ月']
```
