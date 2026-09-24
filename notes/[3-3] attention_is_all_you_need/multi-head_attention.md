
[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)


## MultiHeadAttention

### init

``` python 
    # self.q_projection = 
    # self.k_projection = 
    # self.v_projection = 
    # self.out_projection = 
    # self.dropout = 
```

如何确定线性层的尺寸？

$$

MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O \\
where \ head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

$$

$ W_i^Q $、$ W_i^K $、$ W_i^V $ 即对应的 projection，$ W^O $ 为输出前的 projection；
有 $ W_i^Q \in \mathbb{R}^{d_{model} \times d_k} $、$ W_i^K \in \mathbb{R}^{d_{model} \times d_k} $、$ W_i^V \in \mathbb{R}^{d_{model} \times d_v} $、$ W^O \in \mathbb{R}^{hd_v \times d_{model}} $


``` python
'''
n_embd: Dimensionality of embeddings and hidden states
'''
```

所以 `n_embd` 即单个 token 的特征维度，同时对应模型特征维度，即 $ d_{model} $

`self.attn_hidden_dim = n_embd // n_head` 即 $ d_k $


### project_to_query_key_value

#### 如何得到 $ Q $、$ K $、$ V $？

$ Q $、$ K $、$ V $ 实际是从同一输入学习的三种特征，由于在 attention 计算中的位置不同，所以学到的内容也不同；
因此 $ Q $、$ K $、$ V $ 的生成实际就是三次特征提取，即输入 $ x $ 分别过一遍对应的 linear


#### 多头的引入、形状调整

直接输入 linear 得到的 $ Q $、$ K $、$ V $ 形状均为 `(batch_size, seq_len, n_embd)`，

容易理解的是：head 应该是对 feature volume 的切分（而不是在序列、batch上切），
所以 
``` python
q = q.view(batch, seq_len, n_head, hidden_dim)
```

对于形状的调整，由于是按 head 划分任务，计算时对于整个序列，每组 attention 都只看一个 head 内的序列数据，
所以实际是按 `(n_head, seq_len, hidden_dim)` 的结构取数据计算