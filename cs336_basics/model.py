import torch
from einops import rearrange, einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features:int, out_features:int, weights, device=None, dtype=None):
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.weights = weights
        # no bias for now


    def forward(self, input: torch.Tensor) -> torch.Tensor:

        return einsum(input, self.weights, "batch seq in_features, out_features in_features -> batch seq out_features") 



class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, weights, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_matrix = weights
        

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        # The forward method should select the embedding 
        # vector for each token ID by indexing into an embedding matrix of shape (vocab_size, d_model) using a 
        # torch.LongTensor of token IDs with shape (batch_size, sequence_length)
        
        return self.embedding_matrix[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model : int, eps: float, weights: torch.Tensor, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}

        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.learned_para = weights
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        ssq = einsum(x**2, "... D  -> ...").unsqueeze(-1)
        result = x * self.learned_para / torch.sqrt(ssq/self.d_model + self.eps)
        return result.to(in_dtype)


def SiLU(x):
    return x * x.sigmoid()


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, w1_weight, w2_weight, w3_weight, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight = w1_weight
        self.w2_weight = w2_weight
        self.w3_weight = w3_weight

    # GLU(x, W_1, W_2) = sigmoid(x @ W_1) * (x @ W_2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer1 = Linear(self.d_model, self.d_ff, self.w1_weight)
        layer2 = Linear(self.d_model, self.d_ff, self.w2_weight)
        layer3= Linear(self.d_model, self.d_ff, self.w3_weight)

        return layer2.forward(SiLU(layer1.forward(x)) * layer3.forward(x))