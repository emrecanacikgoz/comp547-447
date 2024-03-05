import torch.nn as nn
import torch.nn.functional as F
import torch


class Attention(nn.Module):
    def __init__(self, head_size, embedding_size, block_size, dropout=0.1):
        """ attention mechanism 
        Args:
            head_size (int): size of each head
            embedding_size (int): size of the input embedding
            block_size (int): size of the block
            dropout (float): dropout rate
        Returns:
            output (tensor): output of the attention mechanism
        """
        super().__init__()

        # TODO-1:
        # Implement key, query, and value linear transformations
        self.key   = None
        self.query = None
        self.value = None

        # used for masking the upper triangular part of the attention matrix
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # TODO-2: Implement the pass for key, query, and value 
        k = None
        q = None
        v = None # (B,T,hs)
        
        # TODO-3: Implement the attention scores computation
        attention = None

        # mask and normalize
        attention = attention.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # TODO-4: get the output, this should be (B, T, hs)
        output = None

        return output
    

class MultiHeadAttention(nn.Module):
    """ 
    multiple heads of self-attention in parallel 
    
    Args:
        num_heads (int): number of heads
        head_size (int): size of each head
        embedding_size (int): size of the input embedding
        block_size (int): size of the block
        dropout (float): dropout rate
    Returns:
        output (tensor): output of the multi-head attention mechanism
    """

    def __init__(self, num_heads, head_size, embedding_size, block_size, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([Attention(head_size, embedding_size, block_size, ) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedFoward(nn.Module):
    """ 
    a linear layer with non-linearity 
    
    Args:
        embedding_size (int): size of the input embedding
        dropout (float): dropout rate
    Returns:
        output (tensor): output of the feed-forward layer
    """

    def __init__(self, embedding_size, dropout=0.1):
        super().__init__()
        # TODO-5: Implement the feed-forward layer
        # This should have two layers of fc with ReLU activation and dropout
        # step1: first layer should take embedding_size and outputs 4*embedding_size
        # step2: add relu non-linearity
        # step3: second layer should take 4*embedding_size and outputs embedding_size
        # step4: add dropout
        self.net = None

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Decoder block

    Args:
        embedding_size (int): size of the input embedding
        num_heads (int): number of heads
        block_size (int): size of the block
    Returns:
        output (tensor): output of the decoder block 
    """

    def __init__(self, embedding_size, num_heads, block_size):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()

        assert embedding_size % num_heads == 0, 'embedding dimension must be divisible by number of heads'

        # TODO-6: Calculate the head size, explain Emre Can why we need to do this by raising hand!
        head_size = None

        self.self_attention = MultiHeadAttention(num_heads, head_size, embedding_size, block_size)
        self.mlp = FeedFoward(embedding_size)
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.self_attention(x)
        x = self.ln2(x)
        x = x + self.mlp(x)
        return x