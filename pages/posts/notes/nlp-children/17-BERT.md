---
title: BERT
date: 2025-07-12
type: notes-nlp
---


_只有编码器的 Transformer 模型_

### BERT 简介

BERT (Bidirectional Encoder Representations from Transformers) 是一种强大的预训练语言模型，它革命性地改变了自然语言处理（NLP）领域。简单来说，BERT 的核心思想是**通过深度学习，让计算机更好地理解文本的上下文含义**。

与之前的模型相比，BERT 的主要创新在于它能够**同时考虑一个词的左边和右边的上下文**来进行双向编码，而此前的模型（如 GPT）通常只考虑单向（从左到右）的上下文。这使得 BERT 在理解词语的多义性方面表现得尤为出色。

下图展示了 BERT 与之前的 ELMo 和 GPT 模型的区别：

![ELMo、GPT 和 BERT 的比较](/images/notes/nlp/elmo-gpt-bert.svg)

- **ELMo**: 虽然也是双向的，但它的架构是为特定任务设计的。
- **GPT**: 是通用的，但只能从左到右读取文本。
- **BERT**: 结合了两者的优点，既是**双向**的，又是**通用**的。

### BERT 的输入表示

为了让模型能够理解输入的文本，BERT 对输入序列进行了特殊处理。一个输入序列可以是一个句子，也可以是两个句子组成的句子对。

BERT 的输入嵌入（Embedding）由三部分相加而成：

1.  **词元嵌入 (Token Embeddings)**：表示每个词或子词本身的含义。
2.  **片段嵌入 (Segment Embeddings)**：用于区分两个不同的句子（例如，在问答任务中区分问题和答案）。
3.  **位置嵌入 (Position Embeddings)**：让模型知道每个词在句子中的位置。

下图清晰地展示了这三种嵌入如何组合成最终的输入表示：

![BERT 输入序列的嵌入](/images/notes/nlp/bert-input.svg)

### BERT 的预训练任务

BERT 的强大能力来自于两个巧妙的预训练任务。模型在海量的文本数据（如维基百科）上完成这两个任务，从而学习到通用的语言知识。

1.  **掩码语言模型 (Masked Language Model, MLM)**
    这个任务是 BERT 实现双向理解的关键。在训练时，系统会随机遮盖（Mask）掉输入句子中 15% 的词，然后让模型去预测这些被遮盖的词是什么。这迫使模型必须根据前后文来推断缺失的词，从而学习到深度的双向语境表示。

    例如，句子 "my dog is hairy" 可能会被处理成 "my dog is [MASKED]"，模型需要根据 "my dog is" 来预测 "hairy"。

2.  **下一句预测 (Next Sentence Prediction, NSP)**
    这个任务让模型学习句子之间的关系。模型会接收一对句子，然后判断第二句是否是第一句在原文中的下一句。

    - 50% 的情况下，第二句确实是下一句。
    - 另外 50% 的情况下，第二句是从语料库中随机挑选的。

    通过这个二元分类任务，BERT 能够更好地理解诸如问答、自然语言推断等需要理解句子间逻辑关系的任务。

---

### BERT 训练 WikiText 详解

#### 输入表示

`get_tokens_and_segments`主要功能是为 BERT 模型的输入准备数据。BERT 模型可以处理单个句子或句子对，这个函数就是用来构建这两种输入格式的。

当输入为单个文本时，BERT 输入序列是特殊类别词元`&lt;cls&gt;`、文本序列的标记、以及特殊分隔词元`&lt;sep&gt;`的连结。当输入为文本对时，BERT 输入序列是`&lt;cls&gt;`、第一个文本序列的标记、`&lt;sep&gt;`、第二个文本序列标记、以及`&lt;sep&gt;`的连结。

```python
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引"""
    # 添加特殊词元"&lt;cls&gt;"到序列开始处，以及"&lt;sep&gt;"到序列A结束处
    tokens = ["&lt;cls&gt;"] + tokens_a + ["&lt;sep&gt;"]
    # 初始化片段索引列表，0表示序列A的部分
    segments = [0] * (len(tokens_a) + 2)
    # 如果提供了第二个序列，则将其添加到tokens中，并更新segments
    if tokens_b is not None:
        tokens += tokens_b + ["&lt;sep&gt;"]
        segments += [1] * (len(tokens_b) + 1)
    # 返回处理后的词元列表和片段索引列表
    return tokens, segments
```

```bash
# 假设这是我们输入的句子A的词元
tokens_a = ['天', '气', '真', '好']

tokens, segments = get_tokens_and_segments(tokens_a)

print("处理后的 Tokens:", tokens)
# 输出: ['&lt;cls&gt;', '天', '气', '真', '好', '&lt;sep&gt;']

print("处理后的 Segments:", segments)
# 输出: [0, 0, 0, 0, 0, 0]
```

1. tokens：在 tokens_a 的开头加上了 `&lt;cls&gt;`，结尾加上了 `&lt;sep&gt;`
2. segments：因为只有一个句子，所以所有的词元都属于第一个片段，其片段索引全部为 0。列表长度与 tokens 相同。

```bash
# 句子A的词元
tokens_a = ['我', '爱', '北京']
# 句子B的词元
tokens_b = ['天安门']

tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)

print("处理后的 Tokens:", tokens)
# 输出: ['&lt;cls&gt;', '我', '爱', '北京', '&lt;sep&gt;', '天安门', '&lt;sep&gt;']

print("处理后的 Segments:", segments)
# 输出: [0, 0, 0, 0, 0, 1, 1]
```

> 1. tokens：
>    - 首先，像单句子一样处理 tokens_a，得到 [`&lt;cls&gt;`, `我`, `爱`, `北京`, `&lt;sep&gt;`]。
>    - 然后，将 tokens_b 和另一个 `&lt;sep&gt;` 拼接到后面。
> 2. segments：
>    - 第一个句子的所有词元（包括 `&lt;cls&gt;` 和第一个 `&lt;sep&gt;`）的片段索引都是 0。
>    - 第二个句子的所有词元（包括它后面的 `&lt;sep&gt;`）的片段索引都是 1。这能帮助模型区分两个不同的句子。

残差连接后进行层规范化，该模块主要用于在前一层的输出上添加残差连接，并在相加后进行层规范化处理。层规范化有助于解决内部协变量偏移问题，加速模型训练过程的收敛。

```python
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)  # 调用父类的构造函数
        self.dropout = nn.Dropout(dropout)  # dropout层，用于防止过拟合
        self.ln = nn.LayerNorm(normalized_shape)  # 层规范化层

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)  # 添加残差连接和层规范化
```

基于位置的前馈网络，该网络对输入进行变换，但不改变其时间步长（序列长度），仅在特征维度上进行操作。主要用于 Transformer 模型中，对注意力机制的输出进行进一步处理。

```python
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)  # 调用父类的构造函数
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)  # 第一层线性变换
        self.relu = nn.ReLU()  # 激活函数ReLU
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)  # 第二层线性变换

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))  # 前馈网络的前向传播
```

`transpose_qkv` 是在 Transformer 的多头注意力（Multi-Head Attention）机制中一个至关重要的辅助函数。它的核心作用是 **改变输入张量（Query, Key, 或 Value）的形状，以便能够高效地进行并行计算**。

```python
def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1) # 将num_hiddens分成num_heads个部分

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3) # 将num_heads维度移到第二个位置

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3]) # 将batch_size和num_heads合并为一个维度
```

```bash
# 假设我们有以下参数
batch_size = 2      # 2个样本
sequence_len = 10   # 每个样本10个词元
num_hiddens = 128   # 每个词元的嵌入维度
num_heads = 8       # 8个注意力头

# 创建一个模拟的输入张量 X (可以是Q, K, 或 V)
X = torch.randn(batch_size, sequence_len, num_hiddens)
原始形状: torch.Size([2, 10, 128])

# 调用函数进行变换
output = transpose_qkv(X, num_heads)

# 1. 分割
step1 = X.reshape(batch_size, sequence_len, num_heads, -1)
1. 分割后形状: torch.Size([2, 10, 8, 16])

# 2. 重排
step2 = step1.permute(0, 2, 1, 3)
2. 重排后形状: torch.Size([2, 8, 10, 16])

# 3. 合并批次
step3 = step2.reshape(-1, sequence_len, 16)
3. 合并批次后形状: torch.Size([16, 10, 16])

最终输出形状: torch.Size([16, 10, 16])
验证: batch_size * num_heads = 16
```

在自然语言处理中，一个批次里的句子（序列）长度通常是不同的。为了能将它们放入一个统一的张量中进行计算，我们通常会用一个特殊的“填充”（padding）词元将所有短句子补齐到和最长的句子一样的长度。

但是，在后续的计算（比如注意力机制）中，我们不希望模型关注这些填充的部分。`sequence_mask` 的作用就是将这些填充位置的值替换成一个特定的值，从而在计算中“屏蔽”或“忽略”它们。

```python
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = (
        torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :]
        < valid_len[:, None]
    )  # 创建一个布尔掩码，标记有效长度的项
    X[~mask] = value  # 将不相关的项设置为指定的值（默认为0）
    return X
```

```bash
# 1. 创建一个模拟的输入张量 X, 假设所有位置的初始值都是 1
X = torch.ones(2, 5)  # batch_size=2, max_len=5

# 2. 定义每个序列的有效长度
valid_len = torch.tensor([3, 2])  # 表示第一个序列有效长度为3，第二个序列有效长度为2
原始张量 X:
tensor([[1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.]])

# 3. 调用函数进行屏蔽，用 -1 替换填充部分
masked_X = sequence_mask(X, valid_len, value=-1)
屏蔽后的张量 X:
tensor([[ 1.,  1.,  1., -1., -1.],
        [ 1.,  1., -1., -1., -1.]])
```

这个 `masked_softmax` 函数是注意力机制中的一个关键组件。它的作用是在计算 Softmax 概率分布时，忽略掉序列中那些无效的填充（padding）部分。

为什么需要它？ 在处理一批句子时，由于句子长度不同，我们通常会用填充符（&lt;pad&gt;）将短句子补齐。在计算注意力权重时，我们不希望模型对这些无意义的填充符分配任何注意力。`masked_softmax` 就是解决这个问题的。

```python
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)  # 对最后一个轴执行softmax
    else:
        shape = X.shape  # 保存X的原始形状
        if valid_lens.dim() == 1:  # 如果valid_lens是1D张量
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])  # 扩展为2D张量
        else:
            valid_lens = valid_lens.reshape(-1)  # 2D张量转1D张量
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)  # 对最后一个轴执行softmax
```

```bash
# 1. 假设这是计算出的注意力分数 (batch_size=2, seq_len=4)
attention_scores = torch.tensor([[1.0, 2.0, 5.0, 4.0],
                                 [3.0, 1.0, 4.0, 2.0]])
# 2. 定义有效长度
valid_lens = torch.tensor([2, 3])

原始注意力分数:
tensor([[1., 2., 5., 4.],
        [3., 1., 4., 2.]])

# 3. 调用 masked_softmax
attention_weights = masked_softmax(attention_scores, valid_lens)

计算出的注意力权重:
tensor([[0.2689, 0.7311, 0.0000, 0.0000],
        [0.2595, 0.0351, 0.7054, 0.0000]])

权重和: # attention_weights.sum(axis=1)
tensor([1.0000, 1.0000])
```

`DotProductAttention`实现的是缩放点积注意力（Scaled Dot-Product Attention），这是 Transformer 最核心的计算单元。它的作用是根据一个“查询”（Query）和一系列“键”（Key），计算出对一系列“值”（Value）的加权平均。

简单来说，它回答了这样一个问题：“当我处理当前这个词（Query）时，我应该对句子中其他所有词（Keys）给予多大的关注度（权重），然后根据这些关注度来融合它们的信息（Values）？”

```python
class DotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)  # 缩放点积
        self.attention_weights = masked_softmax(scores, valid_lens)  # 掩蔽softmax
        return torch.bmm(self.dropout(self.attention_weights), values)  # # 返回加权和
```

```bash
# 1. 初始化注意力层
attention = DotProductAttention(dropout=0.0)
# 关闭 dropout
attention.eval()

# 2. 创建模拟数据 (batch_size, 序列长度, 维度)
# 假设 Q, K, V 都是同一个输入序列（自注意力）
queries = torch.randn(1, 3, 4) # 1个样本, 3个词, 4维向量
keys = queries
values = queries

# 假设序列的有效长度是 2，第3个词是填充
valid_lens = torch.tensor([2])

Queries (Keys, Values) 形状: torch.Size([1, 3, 4]) # queries.shape
有效长度: tensor([2]) # valid_lens


# 3. 前向传播
output = attention(queries, keys, values, valid_lens)

计算出的注意力权重 (形状: torch.Size([1, 3, 3])): # attention.attention_weights.shape 随机值，但结构是固定的
tensor([[[0.4073, 0.5927, 0.0000],
         [0.5488, 0.4512, 0.0000],
         [0.4991, 0.5009, 0.0000]]])

最终输出 (形状: torch.Size([1, 3, 4])): # output.shape 随机值，但结构是固定的
tensor([[[-0.1118,  0.1311,  0.2452, -0.7797],
         [-0.1332,  0.1082,  0.2429, -0.7516],
         [-0.1227,  0.1194,  0.2440, -0.7653]]])
```

`transpose_output` 函数是多头注意力（Multi-Head Attention）机制中的一个“收尾”步骤。它的功能与我们之前讨论的 `transpose_qkv` 函数正好相反。在所有注意力头并行计算完各自的结果后，我们需要将这些分散的结果合并起来，恢复成原始的张量结构。这个函数就是用来完成这个“合并”和“重排”的逆操作。

```python
def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2]) # 将batch_size和num_heads分离
    X = X.permute(0, 2, 1, 3) # 将num_heads维度移到第二个位置
    return X.reshape(X.shape[0], X.shape[1], -1) # 将最后的形状调整为(batch_size, 查询或者“键－值”对的个数, num_hiddens)
```

```bash
# 假设我们有以下参数
batch_size = 2      # 2个样本
sequence_len = 10   # 每个样本10个词元
num_heads = 8       # 8个注意力头
head_dim = 16       # 每个头的维度

原始的总维度 num_hiddens = num_heads * head_dim = 8 * 16 = 128

# 创建一个模拟的、已经过并行注意力计算的输入张量
# 形状为 (batch_size * num_heads, sequence_len, head_dim)
attention_output = torch.randn(batch_size * num_heads, sequence_len, head_dim)
输入形状 (注意力计算后): torch.Size([16, 10, 16])

# 调用函数进行逆向变换
final_output = transpose_output(attention_output, num_heads)

# 1. 还原批次
step1 = attention_output.reshape(batch_size, num_heads, sequence_len, head_dim)
1. 还原批次后形状: torch.Size([2, 8, 10, 16])

# 2. 重排
step2 = step1.permute(0, 2, 1, 3)
2. 重排后形状: torch.Size([2, 10, 8, 16])

# 3. 合并头
step3 = step2.reshape(batch_size, sequence_len, -1)
3. 合并头后形状: torch.Size([2, 10, 128])

最终输出形状: torch.Size([2, 10, 128]) # final_output.shape
验证: 原始 num_hiddens = 128 # num_heads * head_dim
```

`MultiHeadAttention` 的代码实现了 **多头注意力（Multi-Head Attention）** 机制，它是 Transformer 模型成功的关键之一。

核心思想： 与其只用一组 `Query, Key, Value (Q, K, V)` 计算一次注意力，不如将 Q, K, V 分别投影（变换）到多个不同的、更低维度的子空间中，在每个子空间里并行地计算注意力，最后再将所有子空间的结果合并起来。

这就好比让模型从多个不同的“角度”（即“头”）去审视和理解序列中的关系。有的头可能关注语法关系，有的头可能关注词义关联，等等。这使得模型能够捕捉到更丰富、更多样的信息。

```python
class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, num_hiddens, num_heads, dropout, use_bias=False,**kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads  # 头数
        self.attention = DotProductAttention(dropout)  # 缩放点积注意力
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)  # 查询的线性变换
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)  # 键的线性变换
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)  # 值的线性变换
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)  # 输出的线性变换

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0
            )

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
```

`EncoderBlock` 是构成整个 BERT 编码器的核心基础单元。一个完整的 BERT 编码器就是由多个这样的 EncoderBlock 堆叠而成的。
这个模块主要由两个关键的子层组成：

- 多头自注意力 (Multi-Head Self-Attention)
- 基于位置的前馈网络 (Position-wise Feed-Forward Network, FFN)

并且，每个子层的周围都包裹着一个 “Add & Norm” 操作，即残差连接 (Residual Connection) 和 层规范化 (Layer Normalization)。

```python
class EncoderBlock(nn.Module):
    """Transformer编码器块"""

    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias
        ) # 多头注意力
        self.addnorm1 = AddNorm(norm_shape, dropout) # 添加残差连接和层规范化
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens) # 前馈网络
        self.addnorm2 = AddNorm(norm_shape, dropout) # 添加残差连接和层规范化

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens)) # 添加残差连接和层规范化
        return self.addnorm2(Y, self.ffn(Y)) # 添加残差连接和层规范化
```

`BERTEncoder` 是 BERT 模型的核心部分，它负责将输入的文本序列（词元）转换成富含上下文信息的向量表示。它主要由三部分构成：

- 输入嵌入层 (Input Embedding Layer)：将输入的离散词元转换为连续的向量。
- 位置嵌入 (Positional Embedding)：为模型提供序列中词元的位置信息。
- 编码器块堆栈 (Stack of Encoder Blocks)：由多个我们之前讨论过的 EncoderBlock 堆叠而成，是模型学习上下文关系的主要场所。

```python
# @save
class BERTEncoder(nn.Module):
    """BERT编码器"""

    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_layers, dropout, max_len=1000, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)  # 词元嵌入
        self.segment_embedding = nn.Embedding(2, num_hiddens)  # 片段嵌入
        self.blks = nn.Sequential()  # 创建一个顺序容器来存储多个编码器块
        for i in range(num_layers):
            self.blks.add_module(f"{i}",EncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout, True))# 添加编码器块到容器中
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        X = self.token_embedding(tokens) + self.segment_embedding(segments)  #
        X = X + self.pos_embedding.data[:, : X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

```python
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4  # 输入参数
norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2  # 输入参数
encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_layers, dropout)  # 创建BERT编码器实例
tokens = torch.randint(0, vocab_size, (2, 8)) # 创建2个样本，每个样本8个词
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]]) # 创建片段索引
encoded_X = encoder(tokens, segments, None) # 通过编码器获取编码后的输出
encoded_X.shape # 输出编码后的输出的形状
```

```bash
torch.Size([2, 8, 768]) # 输出编码后的输出的形状. 768是BERT的隐藏层维度
```

#### 预训练任务

掩蔽语言模型`MaskLM (Masked Language Model)` 是 BERT 预训练的两个核心任务之一。它的目标是：根据上下文，预测出被遮盖（masked）的词元原本是什么。

这个模块的作用就是接收 BERT 编码器 (BERTEncoder) 的输出，并对指定位置的词元进行预测。

```python
# @save
class MaskLM(nn.Module):
    """BERT的掩蔽语言模型任务"""

    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),  # 输入层到隐藏层的线性变换
            nn.ReLU(),  # 激活函数ReLU
            nn.LayerNorm(num_hiddens),  # 层归一化
            nn.Linear(num_hiddens, vocab_size),  # 输出层到词汇表大小的线性变换
        )

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]  # 预测位置的数量
        pred_positions = pred_positions.reshape(-1)  # 将预测位置展平为一维
        batch_size = X.shape[0]  # 批量大小
        batch_idx = torch.arange(0, batch_size)  # 创建批量索引
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1,1]）
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)  # 重复批量索引以匹配预测位置的数量
        masked_X = X[batch_idx, pred_positions]  # 获取被掩蔽的输入
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))  # 重塑
        mlm_Y_hat = self.mlp(masked_X)  # 通过MLP预测掩蔽的输入
        return mlm_Y_hat  # 返回预测结果
```

```python
mlm = MaskLM(vocab_size, num_hiddens)  # 创建MLM实例
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])  # 创建预测位置
mlm_Y_hat = mlm(encoded_X, mlm_positions)  #
mlm_Y_hat.shape  # 输出预测结果的形状 torch.Size([2, 3, 10000])
```

```python
mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]]) # 创建真实标签
loss = nn.CrossEntropyLoss(reduction="none") # 创建损失函数
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1)) # 计算损失
mlm_l.shape # 输出损失的形状 torch.Size([6])
```

下一句预测`NextSentencePred (Next Sentence Prediction, NSP) `是 BERT 预训练的另一个核心任务。它的目标非常直接：判断输入的两个句子（句子 A 和句子 B）在原文中是否是连续的。

这个任务能让 BERT 模型学习到句子级别的关系，这对于问答（Question Answering）、自然语言推断（Natural Language Inference）等下游任务至关重要。

```python
class NextSentencePred(nn.Module):
    """BERT的下一句预测任务"""

    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)  # 输出层，预测下一句是否为真实的下一句

    def forward(self, X):
        # X的形状：(batchsize,num_hiddens)
        return self.output(X)
```

```python
encoded_X = torch.flatten(encoded_X, start_dim=1)  # 将编码后的结果展平
# NSP的输入形状:(batchsize，num_hiddens)
nsp = NextSentencePred(encoded_X.shape[-1])  # 创建一个NextSentencePred实例
nsp_Y_hat = nsp(encoded_X)  # 通过NextSentencePred获取预测结果
nsp_Y_hat.shape  # 输出预测结果的形状 torch.Size([2, 2])
```

```python
nsp_y = torch.tensor([0, 1])  # 创建真实标签
nsp_l = loss(nsp_Y_hat, nsp_y)  # 计算损失
nsp_l.shape  # 输出损失的形状 torch.Size([2])
```

`BERTModel` 类是将所有部分——BERTEncoder、MaskLM (掩码语言模型) 和 NextSentencePred (下一句预测)——组装在一起的最终成品。它代表了完整的 BERT 预训练模型架构。

```python
# @save
class BERTModel(nn.Module):
    """BERT模型"""

    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        """初始化BERT模型的参数

        参数:
        vocab_size (int): 词汇表大小
        num_hiddens (int): 隐藏层大小
        norm_shape (tuple): LayerNorm的输入形状
        ffn_num_input (int): 前馈网络的输入大小
        ffn_num_hiddens (int): 前馈网络的隐藏层大小
        num_heads (int): 注意力头的数量
        num_layers (int): 编码器层数
        dropout (float): Dropout的概率
        max_len (int): 最大序列长度，默认为1000
        key_size (int): key的大小，默认为768
        query_size (int): query的大小，默认为768
        value_size (int): value的大小，默认为768
        hid_in_features (int): 输入到隐藏层的特征数
        mlm_in_features (int): 输入到掩码语言模型的特征数
        nsp_in_features (int): 输入到下一句预测模型的特征数
        """
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size) # BERT编码器
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens), nn.Tanh()) # 隐藏层
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        """前向传播函数

        参数:
        tokens (torch.Tensor): 输入的token序列
        segments (torch.Tensor): 输入的segment序列
        valid_lens (torch.Tensor): 输入的有效长度序列，可选
        pred_positions (torch.Tensor): 需要预测的位置序列，可选
        """
        encoded_X = self.encoder(tokens, segments, valid_lens) # 通过编码器获取编码后的输出
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions) # 通过掩码语言模型获取预测结果
        else:
            mlm_Y_hat = None # 如果没有预测位置，则返回None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“&lt;cls&gt;”标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat # 返回编码后的序列、掩码语言模型的输出和下一句预测模型的输出
```
