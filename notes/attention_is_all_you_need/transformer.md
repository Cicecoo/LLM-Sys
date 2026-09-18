
### Attention 中 Tensor 的形状

考虑

$$
Q: (batch, n_{token}, dim_{k}) \\

K: (batch, n_{kv}, dim_{k}) 
$$

所以

$$
Q  K^T: (batch, n_{token}, n_{kv})\\
$$

而 $ softmax $ 不改变形状，考虑

$$
V: (batch, n_{kv}, dim_{v})
$$

得到

$$
attn = softmax(\frac{QK^T}{\sqrt{dim_{k}}}) V : (batch, n_{token}, dim_{v})
$$

其中 $ dim_{k} $ 不直接等于 embedding dim，因为 embedding 后往往要先做投影，引入一层可学习的变换；

$Q$ 的尺寸 $n_{token}$ 是由输入序列长度（token 数）决定的，而 $K$、$V$ 的尺寸 $n_{kv}$ 是由已有信息的长度决定的：
对于 self-attention，可查询的就是输入序列本身，所以有 $n_{token} = n_{kv}$；

而 $dim_{v}$ 则是人为设置的特征维度超参数

### Attention 的选择

> **Additive OR Dot-product**: 
> Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.


### Embedding

论文只说用了“可学习的 embidding”， [annotated-transformer](https://nlp.seas.harvard.edu/annotated-transformer/) 中
则直接解释为 `nn.Embedding`；

`nn.Embedding` 本质 `nn.LookupTable`，[参考]() 

**TODO**

在可学习的 embedding 的基础上，还要加上 attention 计算中的可学习的投影，
每个 attention layer 有自己的投影矩阵，投影部分实现为 attention layer 的一部分；
但并不显示进行计算，而是合并在 linear 中, [参考]()

#### 对于 homework3

`llmsys_hw3/minitorch/modules_basic.py` 中的 embedding 先将 word 编码为 one-hot 向量，
`num_embeddings` 实际就是词表大小；
然后再用一个参数矩阵投影到 embedding 空间，从而引入可学习参数；
即使是可学习的 embedding，也需要先把词输入，此处就是以 one-hot 的形式。

one-hot 乘上投影矩阵实际是查表，

$$

[0, 0, 1, 0] \times \begin{bmatrix} e_1 \\ e_2 \\ e_3 \\ e_4 \end{bmatrix} = e_3

$$

词间关系则在训练中得到，如不同 $e$ 之间的距离