# convert_homophone_dict.py
# 用于统一同义词自定义词库的格式
# 拼音2 字b1 字b2 ... 字bk（\t隔开）

input_file = 'resources/Homophone.txt'
output_file = 'resources/Homophone_tab.txt'

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

converted = []
for line in lines:
    parts = line.strip().split()
    if parts:
        converted.append('\t'.join(parts))

with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(converted))

print(f"已生成制表符分隔的词库: {output_file}")