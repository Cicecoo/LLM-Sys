"""
For additional transformer related

Sequential
Embedding

"""
import numpy as np

from .module import Module, Parameter
from .tensor_functions import (zeros, ones, rand, tensor, tensor_from_numpy, zeros_tensor_from_numpy, ones_tensor_from_numpy)
from .nn import one_hot
from .tensor_ops import TensorBackend
from .tensor import Tensor

from typing import Any, Dict, Optional, Sequence, Tuple


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weights : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings # Vocab size
        self.embedding_dim  = embedding_dim  # Embedding Dimension
        ### BEGIN ASSIGN3_2
        # one-hot 向量并不需要自己构造，只是 W 需要适应其形状；
        # 好像还是需要自己构造？forward 中
        self.weights = Parameter(tensor_from_numpy(
            np.random.randn(num_embeddings, embedding_dim),  # TODO：是这个意思吗？
            backend=self.backend,
            requires_grad=True  # ?
        ))
        # raise NotImplementedError
        ### END ASSIGN3_2
    
    def forward(self, x: Tensor):
        """Maps word indices to one-hot vectors, and projects to embedding vectors.

        Args:
            x : Tensor of shape (batch_size, seq_len)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        ### BEGIN ASSIGN3_2
        # v_shape = list(x.shape)
        # one_hot = tensor_from_numpy(np.zeros(v_shape.append(self.num_embeddings)))
        # one_hot = tensor_from_numpy(np.zeros(bs, seq_len, self.num_embeddings))
        # one_hot = tensor_from_numpy(
        #     np.eye(self.num_embeddings)[x] 
        #     )
        
        # 框架已提供 one_hot()
        v = one_hot(x, self.num_embeddings) 
        # v: (bs, seq_len, num_embeddings)
        
        # v 形状为 (bs, seq_len, num_embeddings)
        # weights 形状为 (num_embeddings, embedding_dim) -> (1, num_embeddings, embedding_dim)
        # v @ self.weights.value 矩阵乘 assert 不通过 TODO: 是矩阵乘实现有问题吗？
        
        # 先合并前两维
        v = v.contiguous().view(bs*seq_len, self.num_embeddings)
        # v: (bs*seq_len, num_embeddings)
        
        # 这样就绕过广播了
        output = v @ self.weights.value
        
        return output.view(bs, seq_len, self.embedding_dim)
        # raise NotImplementedError
        ### END ASSIGN3_2

    
class Dropout(Module):
    def __init__(self, p_dropout: float=0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes: 
            p_dropout : Probability an element will be zeroed.
        """
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor: 
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)
        
        Args: 
            x : Tensor of shape (*)
        
        Returns: 
            output : Tensor of shape (*)

        Note: If p_dropout is 0, directly return the input tensor. Otherwise, the random seed may cause problems
        """
        ### BEGIN ASSIGN3_2
        # if self.p_dropout == 0:
        #     return x
        if self.p_dropout == 0 or not self.training:    # "During training"
            return x
        
        # mask = rand(x.shape, backend=x.backend) 
        # 跟测试一致的话需要 np.random
        mask = tensor_from_numpy(
            np.random.rand(*x.shape),   # x.shape 前加 * 解包元素
            backend=x.backend
        )
        
        return x * (mask > self.p_dropout) / (1-self.p_dropout) # TODO: 会不会除0？
        
        # raise NotImplementedError
        ### END ASSIGN3_2


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool, backend: TensorBackend):
        super().__init__()
        """Applies a linear transformation to the incoming data. (Same as PyTorch)

        Parameters:
            in_size  - The size of the dimension the transformation will be applied to
            out_size - The size of the resulting transformation's dimension
            bias     - If True, then add an additive bias

        Attributes:
            weights - The learnable weights of shape (in_size, out_size) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
            bias   - The learnable weights of shape (out_size, ) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
        """
        self.out_size = out_size
        ### BEGIN ASSIGN3_2
       
        # Uniform(-1/sqrt(in_size), 1/sqrt(in_size)) TODO: 这是怎么来的？
        # self.in_size = in_size    需要不需要，为什么
        self.use_bias = bias
        # self.backend = backend
        
        self.weights = Parameter(tensor_from_numpy(
            (np.random.rand(in_size, self.out_size)-0.5)/np.sqrt(in_size),   # 低效？
            backend,
            requires_grad=True
        ))

        if self.use_bias:
            self.bias = Parameter(tensor_from_numpy(
                (np.random.rand(out_size, )-0.5)/np.sqrt(in_size),
                backend,
                requires_grad=True
            ))
        
        # raise NotImplementedError
        ### END ASSIGN3_2

    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data.
        
        Args: 
            x : Tensor of shape (n, in_size)
        
        Returns:
            output : Tensor of shape (n, out_size)
        """
        batch, in_size = x.shape
        ### BEGIN ASSIGN3_2

        output = x @ self.weights.value
        
        if self.use_bias:
            output += self.bias.value
        
        return output

        # raise NotImplementedError
        ### END ASSIGN3_2


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float, backend: TensorBackend):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.
        
        Args: 
            dim : Expected size of the last dimension to apply layer normalization.
            eps : A value added for numerical stability.
        
        Attributes: 
            weights : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias    : the learnable bias of the module of shape (self.dim, ) initialized to 0.
        """
        self.dim = dim
        self.eps = eps
        ### BEGIN ASSIGN3_2
        
        # self.weights = ones((self.dim, ), backend=backend)
        # self.bias = zeros((self.dim, ), backend=backend)
        # 可学习参数必须要包装
        self.weights = Parameter(ones((self.dim, ), backend=backend))
        self.bias = Parameter(zeros((self.dim, ), backend=backend))
        
        # raise NotImplementedError
        ### END ASSIGN3_2

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs. 
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.
        
        Input: 
            x - Tensor of shape (bs, dim)
        
        Output: 
            output - Tensor of shape (bs, dim)
        """
        batch, dim = x.shape
        ### BEGIN ASSIGN3_2
        
        mean = x.mean(dim=1)
        var = x.var(dim=1)
        norm = (x-mean) / (var + self.eps) ** 0.5   # TODO: 没有 sqrt 
        # output = self.weights * norm + self.bias
        output = norm * self.weights.value + self.bias.value    # TODO: 访问方式
        
        return output
        # raise NotImplementedError
        ### END ASSIGN3_2


"""
NOTE
1. @ 和 *; python 的 “运算符重载”
2. embedding 的原理、layernorm 的原理
3. Parameter 封装需更新参数: 框架的实现
"""
