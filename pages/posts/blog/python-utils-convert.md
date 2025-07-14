---
title: NLP Utils: NER 标注数据转换为 BIO 标注格式
date: 2025-07-14
draft: true
lang: zh
---

## BIO 标注格式

- B-X: 表示实体 X 的开头(Begin)
- I-X: 表示实体 X 的内部/延续部分(Inside)
- O: 表示不属于任何实体类型(Outside)

## `get_token()` 函数

功能: 对输入文本进行分词处理

- 将连续的英文字母作为一个 token
- 中文字符每个字作为单独的 token
- 例如："中国 people"会被分为["中","国","people"]

```python
def get_token(input):
    #english = 'abcdefghijklmnopqrstuvwxyz0123456789'
    english = 'abcdefghijklmnopqrstuvwxyz'
    output = []
    buffer = ''
    for s in input:
        if s in english or s in english.upper():
            buffer += s
        else:
            if buffer: output.append(buffer)
            buffer = ''
            output.append(s)
    if buffer: output.append(buffer)
    return output
```

## `json2bio()` 函数

转换过程

- 文本预处理: 将换行符替换为空格
- 分词: 使用 get_token()对文本分词
- 初始化标签: 创建与 token 数量相同的'O'标签列表
- 标注转换:

  - 将起始位置标记为 B-实体类型
  - 将中间位置标记为 I-实体类型

- 输出格式: 每行一个"token 标签"对

```python
def json2bio(fpath,output,splitby = 's'):
    with open(fpath, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            annotations = json.loads(line)
            text = annotations['text'].replace('\n',' ')
            all_words = get_token(text.replace(' ',','))
            all_label = ['O'] * len(all_words)
            for i in annotations['label']:
                b_location = i[0]
                e_location = i[1]
                label = i[2]
                all_label[b_location] = 'B-'+label
                if b_location != e_location:
                    for word in range(b_location+1,e_location):
                        all_label[word] = 'I-'+label
            cur_line = 0
            # 写入文件
            toekn_label = zip(all_words,all_label)
            with open(output,'a',encoding='utf-8') as f:
                for tl in toekn_label:
                    f.write(tl[0]+str(' ')+tl[1])
                    f.write('\n')
                    cur_line += 1
                    if cur_line == len(all_words):
                        f.write('\n') # 空格间隔不同句子


```

输出示例:
对于文本`词汇阅读是关键 08年考研暑期英语复习全指南`，输出可能是：

```bash
词 O
汇 O
阅 O
读 O
是 O
关 O
键 O
  O
08 B-year
年 I-year
考 B-exam
研 I-exam
暑 I-exam
期 I-exam
英 I-exam
语 I-exam
复 O
习 O
全 O
指 O
南 O

```

## 应用场景

1. NER 模型训练数据准备: 将人工标注的 JSON 数据转换为模型可用的 BIO 格式
2. 数据格式标准化: 统一不同标注工具的输出格式
3. 深度学习预处理: 为 BERT 等预训练模型准备输入数据
