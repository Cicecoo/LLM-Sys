

### 模型架构

文中只给出一段描述

> **2.3. Model** \
>  We use a Transformer (Vaswani et al., 2017) based architecture for our LMs. The model largely follows the details of the OpenAI GPT model (Radford et al., 2018) with a  few modifications. Layer normalization (Ba et al., 2016) was moved to the input of each sub-block, similar to a pre-activation residual network (He et al., 2016) and an additional layer normalization was added after the final selfattention block. A modified initialization which accounts for the accumulation on the residual path with model depth is used. We scale the weights of residual layers at initialization by a factor of 1/√N where N is the number of residual layers. The vocabulary is expanded to 50,257. We also increase the context size from 512 to 1024 tokens and a larger batchsize of 512 is used.
> | Parameters | Layers | $d_{model}$ |
> | ---------- | -----: | ----------: |
> | 117M       | 12     | 768         |
> | 345M       | 24     | 1024        |
> | 762M       | 36     | 1280        |
> | 1542M      | 48     | 1600        |


1. 沿用 GPT-1 的 Transformer 架构

2. 将 LayerNorm 移到每个子模块的输入处，即 Pre-LN \
    在自注意力子模块和前馈子模块的输入端进行 LayerNorm。对一个子模块而言，结构可表示为: \
     $y=x+\operatorname{SubBlock}(\operatorname{LN}(x))$ \
    即先对输入归一化，再进行子模块计算，最后与残差分支相加。
    
3. 在模型末端额外增加一次 LayerNorm \
    除了各子模块输入处的 LayerNorm，还在最后一个 Transformer block 之后增加一次 LayerNorm。

4. 根据模型深度调整残差层的权重初始化 \
    初始化时，将残差层的权重缩放为原来的：\
    $ \frac{1}{\sqrt{N}} $ \
    其中，\(N\) 是残差层的数量。目的是考虑模型加深时，残差路径上逐层累加的影响。

5. 词表大小增加到 50,257 个 token 条目。

6. 上下文长度从 512 tokens 增加到 1024 tokens。

7. 使用 batch size = 512，即每个训练批次包含 512 条输入序列。

8. 设置如表的四种模型规模

为了实现 hw3-3，继续参考 [gpt1](../[3-3]%20generative_pre-training/gpt1.md)