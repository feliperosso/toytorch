"""  
nnLayers Module

This module contains an explicit construction of several
of the most common layers used in machine learning:
    - Convolution
    - Dropout
    - Embedding
    - Layer Norm
    - Linear
    - LSTM
    - Multihead Attention
    - Pooling
    - Positional Encoding
    - TransformerDecoderLayer
    - TransformerEncoderLayer

The only torch.nn commands used in this file are:
    - torch.nn.Module
    - torch.nn.Parameters
"""

# Load packages
import torch, math
import numpy as np
# Load functions
from toytorch.func import softmax, relu, sigmoid

# - Convolutional -
class Convolution(torch.nn.Module):
    # Agrees with Pytorch implementation
    def __init__(self, in_channels, out_channels, kernel_size, image_size, stride=1, padding=0):
        super().__init__()
        # Initialize weights
        stdv = math.sqrt(1/(in_channels*kernel_size[0]*kernel_size[1]))
        self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size).uniform_(-stdv, stdv))        
        self.bias = torch.nn.Parameter(torch.empty(out_channels).uniform_(-stdv, stdv))
        # Store Parameters
        self.padding = padding

        """ Construct projector used to expand the size of the input (during forward)
        in order to account for overlaps of the kernel when computing the convolution """ 
        # Note: Instead of constructing this tensor to perform the projection, one could directly
        # do the projection using the "as_strided" command on the input. This would more likely be more efficient.
        n_kernel, m_kernel = kernel_size
        n_in, m_in = image_size
        # Take padding into account
        if padding != 0:
            n_in, m_in = n_in + 2*padding, m_in + 2*padding
        # Dimensions of the output of the convolution
        n_out, m_out = int(math.floor((n_in - n_kernel)//stride + 1)), int(math.floor((m_in - m_kernel)//stride + 1))
        # Create the appropriate projector
        projector = torch.zeros(n_out, m_out, n_kernel, m_kernel, n_in, m_in)
        I, J, n, m = np.ogrid[:n_out, :m_out, :n_kernel, :m_kernel]
        projector[I, J, n, m, I*stride + n, J*stride + m] = 1
        # Store the projector
        self.projector = projector
    
    def forward(self, input):
        """ input: (batch_size, in_channels, height_in, width_out) 
            output: (batch_size, out_channels, height_out, width_out) """
        self.projector = self.projector.to(input.device)
        # Pad the input if required
        if self.padding != 0:
            pad = torch.zeros(*input.shape[:2], input.shape[2] + 2*self.padding,
                               input.shape[3] + 2*self.padding, device=input.device)
            pad[:, :, self.padding:-self.padding, self.padding:-self.padding] = input
            input = pad
        # Apply the input projector to the input
        new_input = torch.einsum('ijkl, mnopkl->ijmnop', input, self.projector)
        # Compute the convolution
        output = torch.einsum('ijklmn, ojmn-> iokl', new_input, self.weight).swapaxes(-1, -3) + self.bias
        return output.swapaxes(-1, -3)

# - Dropout -
class Dropout(torch.nn.Module):
    # Does not agree with Pytorch implementation because of weird random init.
    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate # If dropout_rate = 0, mask == 1.
    
    def forward(self, input):
        if self.training:
            mask = torch.zeros(*input.shape, device=input.device).uniform_(0, 1) > self.dropout_rate
            return input*mask/(1 - self.dropout_rate)
        else:
            return input

# - Embedding - 
class Embedding(torch.nn.Module):
    # Matches with Pytorch implementation
    def __init__(self, vocabulary_size:int, dim_embed:int):
        super().__init__()
        self.embedding_matrix = torch.nn.Parameter(torch.randn(vocabulary_size, dim_embed))
    
    def forward(self, input):
        """ Input: (*)
            Output: (*, dim_embed) """
        return self.embedding_matrix[input, :]

# - Layer Norm -
class LayerNorm(torch.nn.Module):
    # Does not agree with Pytorch implementation by overall factor, as std is computed weirdly in Pytorch.
    def __init__(self, dim_embed:int, eps=1e-5):
        super().__init__()
        # Parameters
        self.dim_embed = dim_embed
        self.eps = eps
        # Initialize parameters
        self.weight = torch.nn.Parameter(torch.ones(dim_embed, requires_grad=True))
        self.bias = torch.nn.Parameter(torch.zeros(dim_embed, requires_grad=True))
    
    def forward(self, input):
        """ Input dim: (*, dim_embed). mean and std computed along embed_dim."""
        std, mean = torch.std_mean(input, correction=0, dim=-1, keepdim=True)
        output = ((input - mean)/torch.sqrt(std + self.eps))*self.weight + self.bias
        return output

# - Linear -
class Linear(torch.nn.Module):
    # Agrees with the Pytorch implementation.
    def __init__(self, dim_in:int, dim_out:int):
        super().__init__()
        stdv = 1/math.sqrt(dim_in)
        self.weights = torch.nn.Parameter(torch.empty(dim_out, dim_in).uniform_(-stdv, stdv))
        self.biases = torch.nn.Parameter(torch.empty(dim_out).uniform_(-stdv, stdv))
        
    def forward(self, input):
        """ Input dim: (*, dim_in)
            Output dim: (*, dim_out) """
        output = input@self.weights.T + self.biases
        return output

# - LSTM -
class LSTM(torch.nn.Module):  
    # Agrees with Pytorch implementation
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Store dimensions
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        # Initialize weights and biases
        stdv = 1/(math.sqrt(hidden_dim))
        self.weight_ih = torch.nn.Parameter(torch.empty(4*hidden_dim, input_dim).uniform_(-stdv, stdv))
        self.weight_hh = torch.nn.Parameter(torch.empty(4*hidden_dim, hidden_dim).uniform_(-stdv, stdv))

        self.bias_ih = torch.nn.Parameter(torch.empty(4*hidden_dim).uniform_(-stdv, stdv))
        self.bias_hh = torch.nn.Parameter(torch.empty(4*hidden_dim).uniform_(-stdv, stdv))
    
    def forward(self, input, h_prev, c_prev):
        """ input: (seq_len, batch_size, input_dim)
            h_prev, c_prev: (batch_size, hidden_dim)
            output: (seq_len, batch_size, hidden_dim)"""
        output = torch.zeros(*input.shape[:-1], self.hidden_dim)
        for num, token in enumerate(input):
            # Apply linear transformations
            linear = (token@self.weight_ih.T + self.bias_ih) + (h_prev@self.weight_hh.T + self.bias_hh)
            # Non-linearities
            i_t = sigmoid(linear[:, 0*self.hidden_dim: 1*self.hidden_dim])
            f_t = sigmoid(linear[:, 1*self.hidden_dim: 2*self.hidden_dim])
            g_t = torch.tanh(linear[:, 2*self.hidden_dim: 3*self.hidden_dim])
            o_t = sigmoid(linear[:, 3*self.hidden_dim: 4*self.hidden_dim])
            # Final combinations
            c_t = f_t*c_prev + i_t*g_t
            h_t = o_t*torch.tanh(c_t)
            # Save hidden states
            output[num] = h_t
            # Rename hidden and cell states
            h_prev, c_prev = h_t, c_t
        return output, (h_prev, c_prev)

# - Multihead Attention -
class MultiheadAttention(torch.nn.Module):
    # Matches the Pytorch implementation
    def __init__(self, dim_model:int, num_heads:int):
        super().__init__()
        # Check divisibility between model dimension and number of heads
        if dim_model % num_heads != 0:
            raise Exception("The model dimension must be divisible by the number of heads")
        # Store values
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_k = dim_model//num_heads
        # Initialize weights:
        stdv_in = math.sqrt(3/(2*dim_model)) # The weird 3/2 is included to match with the Pytorch initialization
        stdv_out = math.sqrt(1/(dim_model))
        self.output_weight = torch.nn.Parameter(torch.empty(dim_model, dim_model).uniform_(-stdv_out, stdv_out).requires_grad_(True))
        self.input_weight = torch.nn.Parameter(torch.empty(3, dim_model, dim_model).uniform_(-stdv_in, stdv_in).requires_grad_(True))
        

    def forward(self, query, key, value, mask=None, mean_simi_across_heads=True):
        """ Input: query, key and value shape (batch_size, sequence_length, dim_embedding).
            Output: attention (batch_size, sequence_length, dim_embedding)
                    similarity (batch_size, num_heads*, sequence_length, sequence_length)
                    * num_heads only if mean_simi_across_heads=False"""
        # Apply projectors and reshape the query, key and value
        queries, keys, values = [(x@weight.T).view(*x.shape[:2], self.num_heads, self.dim_k).transpose(1, 2)
                                for x, weight in zip((query, key, value), self.input_weight)]
        # Compute similarity
        pre_similarity = queries@keys.transpose(-2, -1)/math.sqrt(self.dim_k)
        if mask is not None:
            pre_similarity = pre_similarity + mask
        similarity = softmax(pre_similarity, dim=-1)
        # Compute attention
        attention = (similarity@values).transpose(1, 2)
        attention = attention.reshape(*attention.shape[:-2], -1)
        attention = attention@self.output_weight.T
        # Average sim across heads if required
        if mean_simi_across_heads:
            similarity = torch.mean(similarity, dim=-3)
        return attention, similarity

# - Pooling (Max or Mean) -
class Pooling(torch.nn.Module):
    # Agrees with Pytorch implementation
    def __init__(self, kernel_size, image_size, stride, operation='max'):
        super().__init__()
        if operation != 'max' and operation != 'mean':
            raise Exception("Operation must be 'max' or 'mean'.")
        # Store parameters
        self.operation = operation
        """ Construct projector used to expand the size of the input (during forward)
        in order to account for overlaps of the kernel when computing the convolution """
        n_kernel, m_kernel = kernel_size
        n_in, m_in = image_size
        # Dimensions of the output of the convolution
        n_out, m_out = int(math.floor((n_in - n_kernel)//stride + 1)), int(math.floor((m_in - m_kernel)//stride + 1))
        # Create the appropriate projector
        projector = torch.zeros(n_out, m_out, n_kernel, m_kernel, n_in, m_in)
        I, J, n, m = np.ogrid[:n_out, :m_out, :n_kernel, :m_kernel]
        projector[I, J, n, m, I*stride + n, J*stride + m] = 1
        # Store the projector
        self.projector = projector
    
    def forward(self, input):
        """ input: (batch_size, in_channels, in_height, in_width)
            output: (batch_size, in_channels, out_height, out_width) """
        self.projector = self.projector.to(input.device)
        # Apply the input projector to the input
        new_input = torch.einsum('ijkl, mnopkl->ijmnop', input, self.projector)
        if self.operation == 'max':
            return torch.max(torch.max(new_input, dim=-1)[0], dim=-1)[0]
        else:
            return torch.mean(new_input, dim=(-1, -2))

# - Positional Encoding -
class PositionalEncoding(torch.nn.Module):
    # Agrees with the implementation in "The annotated Transformer".
    def __init__(self, dim_model:int, max_seq_length:int=5000):
        super().__init__()
        # Create positional encoding tensor
        self.positional_encoding = torch.zeros(max_seq_length, dim_model)
        # Create Positional Embedding vector for sequence of max_length
        seq_vector = torch.arange(max_seq_length).unsqueeze(-1)
        even_emb = torch.arange(0, dim_model, 2)
        freq_i = 1/(10000**(even_emb/dim_model)).unsqueeze(0) # Frequencies (embedding space)
        
        self.positional_encoding[:, 0::2] = torch.sin(seq_vector*freq_i)
        self.positional_encoding[:, 1::2] = torch.cos(seq_vector*freq_i)
    
    def forward(self, input):
        """ input: (batch_size, sequence_length, dim_embedding)
            output: (batch_size, sequence_length, dim_embedding) """
        self.positional_encoding = self.positional_encoding.to(input.device)
        output = input + self.positional_encoding[:input.shape[1], :]
        return output

# - Transformer Decoder Layer -
class TransformerDecoderLayer(torch.nn.Module):

    def __init__(self, dim_model, num_heads, dim_ff, dropout_rate):
        super().__init__()
        # First sublayer: Self-Multihead Attention
        self.self_attention_layer = MultiheadAttention(dim_model, num_heads)
        self.layer_norm1 = LayerNorm(dim_model)
        # Second sublayer: Cross-Multihead Attention
        self.cross_attention_layer = MultiheadAttention(dim_model, num_heads)
        self.layer_norm2 = LayerNorm(dim_model)
        # Third sublayer: Position-Wise Feedforward
        self.linear1 = Linear(dim_model, dim_ff)
        self.linear2 = Linear(dim_ff, dim_model)
        self.layer_norm3 = LayerNorm(dim_model)

        self.dropout = Dropout(dropout_rate)
    
    def forward(self, decoder_input, encoder_output, self_mask=None, cross_mask=None):
        """ decoder_input/encoder_output: (batch_size, sequence_length, embedding_dim)
            self_mask/cross_mask: (sequence_length, sequence_length)"""
        # First sublayer
        self_attention, self_similarity = self.self_attention_layer(decoder_input, decoder_input,
                                                               decoder_input, self_mask)
        out1 = self.layer_norm1(self.dropout(self_attention) + decoder_input)
        # Second sublayer
        cross_attention, cross_similarity = self.cross_attention_layer(out1, encoder_output,
                                                                       encoder_output, cross_mask)
        out2 = self.layer_norm2(self.dropout(cross_attention) + out1)
        # Third sublayer
        pre = self.linear2(relu(self.linear1(out2)))
        out3 = self.layer_norm3(self.dropout(pre) + out2)
        return out3, (self_similarity, cross_similarity)

# - Transformer Encoder Layer -
class TransformerEncoderLayer(torch.nn.Module):

    def __init__(self, dim_model, num_heads, dim_ff, dropout_rate):
        super().__init__()
        # First sublayer: Self-Multihead Attention
        self.self_attention_layer = MultiheadAttention(dim_model, num_heads)
        self.layer_norm1 = LayerNorm(dim_model)
        # Second sublayer: Position-Wise Feedforward
        self.linear1 = Linear(dim_model, dim_ff)
        self.linear2 = Linear(dim_ff, dim_model)
        self.layer_norm2 = LayerNorm(dim_model)

        self.dropout = Dropout(dropout_rate)
    
    def forward(self, input, mask=None):
        """ input: (batch_size, sequence_length, embedding_dim)
            mask: (sequence_length, sequence_length)"""
        # First layer
        self_attention, similarity = self.self_attention_layer(input, input, input, mask)
        out1 = self.layer_norm1(self.dropout(self_attention) + input)
        # Second layer
        pre = self.linear2(relu(self.linear1(out1)))
        out2 = self.layer_norm2(self.dropout(pre) + out1)
        return out2, similarity
