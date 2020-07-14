#author: Lu Tongyu

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.data.sampler as sampler

import numpy as np
import math
import random

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.
        Args:
        	q: Queries [B, L_q, D_q]
        	k: Keys [B, L_k, D_k]
        	v: Values [B, L_v, D_v], L_v=L_k
        	scale: float tensor
        	attn_mask: Masking [B, L_q, L_k]
        Returns:
        	context: [B, L_q, D_v]
            attetention: [B, L_q, L_k]
        """
        # attention: [B, L_q, L_k]
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale!=None:
        	attention = attention * scale
        #print("******q:",q.shape)
        #print("******k:",k.shape)
        #print("******attention:",attention.shape)
        #print("******v:",v.shape)
        if attn_mask!=None:
        	attention = attention.masked_fill_(attn_mask, -np.inf)
		# softmax
        attention = self.softmax(attention)
		# dropout
        attention = self.dropout(attention)
		# dot product with V
        # context: [B, L_q, D_v]
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        """
        Args:
        	q: Queries [B, L_q, D_q]
        	k: Keys [B, L_k, D_k]
        	v: Values [B, L_v, D_v]
        	attn_mask: Masking [B, L_q, L_k]
        Returns:
        	output:[B, Lq, Dv]
            attention: [B*h, Lq, Lk]
        """
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        # shape = [B*h, L_k/v/q, D/h]
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask!=None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        if (key.size(-1) // num_heads)<=0:
            scale = 0.5
        else:
            scale = (key.size(-1) // num_heads)
        # context [B*h, Lq, Dv/h]
        # attention [B*h, Lq, Lk]
        context, attention = self.dot_product_attention(
          query, key, value, scale, attn_mask)

        # concat heads
        # context [B, Lq, Dv]
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


def residual(sublayer_fn,x):
	return sublayer_fn(x)+x


def padding_mask(seq_k, seq_q):
	# seq_k or seq_q: [B,Lk] or [B,Lq]
    len_q = seq_q.size(1)
    # `auto_PAD` is 0
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1).to(device)  # shape [B, L_q, L_k]
    return pad_mask

def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                    diagonal=1).to(device)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_seq_len):
        """初始化。
        Args:
            d_model: 一个标量。模型的维度，论文默认是512
            max_seq_len: 一个标量。文本序列的最大长度
        """
        global device
        super(PositionalEncoding, self).__init__()
        # PE matrix
        position_encoding = torch.FloatTensor([
          [pos / pow(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
          for pos in range(max_seq_len)]).to(device)
        # sin for even，cos for odd
        position_encoding[:, 0::2] = torch.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = torch.cos(position_encoding[:, 1::2])

        # first line of PE: `auto_PAD` positional encoding，
        pad_row = torch.zeros([1, d_model]).to(device)
        position_encoding = torch.cat((pad_row, position_encoding))

        self.position_encoding = nn.Embedding(max_seq_len, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False).to(device)
    def forward(self, input_len):
        """
        Args:
          input_len: [B, 1]; contents: L
        Returns:
          [B, L, D]
        """

        max_len = torch.max(input_len)
        input_pos = torch.LongTensor(
            [list(range(1, len+1)) + [0] * (max_len - len) for len in input_len]).to(device)
        # range from 1: avoid PAD(0)

        return self.position_encoding(input_pos)


class EmbeddingLayer(nn.Module):
    def __init__(self,config,vocab_size,embedding_init=None):
        super(EmbeddingLayer, self).__init__()
        self.max_seq_len = config.max_output_len
        self.model_dim = config.emb_dim

        self.seq_embedding = nn.Embedding(vocab_size, self.model_dim, padding_idx=0)
        self.token_type_embedding = nn.Embedding(vocab_size, self.model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(self.model_dim, self.max_seq_len)

        nn.init.orthogonal_(self.seq_embedding.weight)
        # nn.init.orthogonal_(self.position_embeddings.weight)
        nn.init.orthogonal_(self.token_type_embedding.weight)
        epsilon = 1e-8
        self.seq_embedding.weight.data = self.seq_embedding.weight.data.div(
            torch.norm(self.seq_embedding.weight, p=2, dim=1, keepdim=True).data+epsilon).to(device)
        self.token_type_embedding.weight.data = self.token_type_embedding.weight.data.div(
            torch.norm(self.token_type_embedding.weight, p=2, dim=1, keepdim=True).data+epsilon).to(device)
        if embedding_init != None:
            self.seq_embedding.weight = torch.nn.Parameter(embedding_init).to(device)

    def forward(self,inputs,input_token_ids,inputs_len):
        # inputs: [B,Le]
        seq_emb = self.seq_embedding(inputs)
        token_emb = self.token_type_embedding(input_token_ids)
        pos_emb = self.pos_embedding(inputs_len)
        output = seq_emb+token_emb+pos_emb
        #print('seq_emb:',seq_emb.shape)
        #print('token_emb:',token_emb.shape)
        #print('pos_emb:',pos_emb.shape)
        return output

class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        #x: [B, L, D]
        #output: [B, D, L]
        output = x.transpose(1, 2)
        #output: [B, D, L]->[B, ffn_dim, L]->[B, D, L]
        output = self.w2(F.relu(self.w1(output)))
        #output: [B, L, D]
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output

class EncoderLayer(nn.Module):
    """Encoder的一层。"""
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        # self attention
        # context: [B, Le, De]
        # attention: [B*h, Le, Le]
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        
        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class TransformerEncoder(nn.Module):
    """多层EncoderLayer组成Encoder。"""

    def __init__(self,config):
        super(TransformerEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList(
          [EncoderLayer(config.emb_dim, config.num_heads, config.ffn_dim,config.dropout) for _ in
           range(config.n_layers)])

    def forward(self, vec_inputs,self_attention_mask=None):
        # vec_inputs: [B,Le,De]
        # output: [B,Le,De]
        output = vec_inputs
        attentions = []
        for encoder in self.encoder_layers:
            # output: [B, Le, De]
            # attention: [B*h, Le, Le]
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)

        return output, attentions


class VectorQuantizer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = vocab_size
        # self._embedding.weight: [V,D]
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim).to(device)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        # commitment_cost: beta
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # inputs: [B,L,D]
        input_shape = inputs.shape
        
        # Flatten input: [B*L,D]
        flat_input = inputs.view(-1, self._embedding_dim)
        #flat_input = inputs
        
        # Calculate distances: d(a,b)=a2+b2-2ab
        # distances: [B*L,V]
        a2 = torch.sum(flat_input**2, dim=1, keepdim=True)#[B*L,1]
        b2 = torch.sum(self._embedding.weight**2, dim=1)#[V]
        ab = 2 * torch.matmul(flat_input, self._embedding.weight.t())#[B*L,V]
        distances = a2+b2-2*ab
            
        # Encoding
        # encoding_indices: [B*L]
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        # encodings: [B*L,V], 1-hot form
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        # quantized: [B,L,D]
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        # encoding_indices: [B,L]
        encoding_indices = encoding_indices.view(input_shape[0:2])
        
        # Loss
        # |sg[zq]-ze|2
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        # |zq-sg[ze]|2
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # zq=ze+sg[zq-ze], for grad straight through
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # loass: float
        # quantized: [B,L,D]
        # perplexity: float
        # encoding_indices: [B,L]
        return quantized, encoding_indices, loss, perplexity


class VectorQuantizerEMA(nn.Module):
    def __init__(self, vocab_size, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = vocab_size
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim).to(device)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(vocab_size))
        self._ema_w = nn.Parameter(torch.Tensor(vocab_size, self._embedding_dim)).to(device)
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # inputs: [B,L,D]
        input_shape = inputs.shape
        
        # Flatten input
        # flat_input: [B*L,D]
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        # distances: [B*L,V]
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        # encoding_indices: [B*L]
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        # encoding: [B*L,V]
        encodings.scatter_(1, encoding_indices, 1)
        # encoding_indices: [B,L]
        encoding_indices = encoding_indices.view(input_shape[0:2])
        # Quantize and unflatten
        # quantized: [B,L,D]
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            # dw: [V,D]
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # loass: float
        # quantized: [B,L,D]
        # perplexity: float
        # encoding_indices: [B,L]
        return quantized, encoding_indices, loss, perplexity


class VQEncoder(nn.Module):
    def __init__(self,config,vocab_size, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VQEncoder, self).__init__()
        self.encoder = TransformerEncoder(config)
        self.linear_mu = nn.Linear(config.emb_dim,config.emb_dim)
        self.linear_logvar = nn.Linear(config.emb_dim,config.emb_dim)
        if decay>0.0+epsilon:
            self.quantizer = VectorQuantizerEMA(vocab_size*config.vq_vocab_size_factor, config.emb_dim, commitment_cost, decay, epsilon)
        else:
            self.quantizer = VectorQuantizer(vocab_size*config.vq_vocab_size_factor, config.emb_dim, commitment_cost)
        
    def forward(self, vec_inputs,self_attention_mask=None,var_factor=0.02):
        ze, attentions = self.encoder(vec_inputs, self_attention_mask)
        mu = ze
        var = 0
        #mu = self.linear_mu(ze)
        #var = torch.exp(self.linear_logvar(ze))
        #ze_sample = torch.mul(torch.randn(ze.shape).to(device),var)*var_factor+mu
        zq, encoding_indices, loss_vq, perplexity_vq = self.quantizer(ze)
        return zq, mu, var, attentions, loss_vq, perplexity_vq, encoding_indices



class EC2Splitter(nn.Module):
    def __init__(self,emb_dim,splitted_dim_list):
        super(EC2Splitter, self).__init__()
        self.emb_dim = emb_dim
        self.splitted_dim_list = splitted_dim_list
        if sum(splitted_dim_list)!=emb_dim:
            assert 0, 'Component dims not compatible with embedding as a whole!'
    def forward(self,emb_in):
        #emb_in: [B,L,D]
        in_shape = emb_in.shape
        if in_shape[2]!=self.emb_dim:
            assert 0, 'Embedding input dim fails!'
        start = 0
        splitted = []
        for segment_len in self.splitted_dim_list:
            end = start+segment_len
            splitted.append(emb_in[:,:,start:end])
            start = end
        return splitted

class TokenClassifier(nn.Module):
    def __init__(self,classify_dim,out_dim):
        super(TokenClassifier, self).__init__()
        self.classify_dim = classify_dim
        self.out_dim = out_dim
        self.classifier = nn.Linear(self.classify_dim, out_dim, bias=True)
    def forward(self,z):
        # z: [B,L,Dc] or [B,Dc]
        # temp, output: [B,L,M] or [B,M]
        output = self.classifier(z)
        # if len(input.shape)==2:
        #     output = F.softmax(temp,dim = 1)
        # elif len(input.shape)==3:
        #     output = F.softmax(temp,dim = 2)
        # else:
        #     assert 0,'dim not compatible!'
        return output


class Pooler(nn.Module):
    def __init__(self, model_dim):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(model_dim, model_dim)
        self.activation = nn.Tanh()
    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class PredictionTransform(nn.Module):
    def __init__(self, model_dim, out_dim):
        super(PredictionTransform, self).__init__()
        self.dense1 = nn.Linear(model_dim, out_dim)
        self.transform_act_fn = gelu
        self.LayerNorm = nn.LayerNorm(out_dim)
    def forward(self, hidden_states):
        hidden_states = self.transform_act_fn(self.dense1(hidden_states))
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class LMPredictionHead(nn.Module):
    def __init__(self, model_dim, embedding_weights):
        super(LMPredictionHead, self).__init__()
        self.transform = PredictionTransform(model_dim, embedding_weights.size(1))
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(embedding_weights.size(1),
                                 embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = embedding_weights
        self.bias = nn.Parameter(torch.zeros(embedding_weights.size(0))).to(device)
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class VQRhythmModel(nn.Module):
    def __init__(self,config,vocab_size,
                 embedding_init=None, commitment_cost=0.25, decay=0, epsilon=1e-5, use_vq = True):
        super(VQRhythmModel, self).__init__()
        #e.g. model_dim = 16
        #     splitted_dim_list = [4,5,7]
        #     classifier_out_len_list = [2,3]
        self.emb_layer = EmbeddingLayer(config,vocab_size,embedding_init)
        self.use_vq = use_vq
        if use_vq:
            self.vq_encoder = VQEncoder(config,vocab_size,commitment_cost,decay,epsilon)
        else:
            self.encoder = TransformerEncoder(config)
        self.splitter = EC2Splitter(config.emb_dim,config.splitted_dim_list)
        self.splitted_dim_list = config.splitted_dim_list
        self.classifier_out_len_list = config.classifier_out_len_list
        if len(config.classifier_out_len_list)>len(config.splitted_dim_list):
            assert 0,'split lengths not compatible!'

        self.classifier_pred = LMPredictionHead(config.emb_dim,self.emb_layer.seq_embedding.weight)
        self.classifier_meter = PredictionTransform(self.splitted_dim_list[0],self.classifier_out_len_list[0])
        self.classifier_grouping = PredictionTransform(self.splitted_dim_list[1],self.classifier_out_len_list[1])
    
    def forward(self,input_sen,input_token_ids,input_len,var_factor=0.02,seq_mask = None):
        # self_attention_mask: [B,Le,Le]
        vec_input = self.emb_layer(input_sen,input_token_ids,input_len)
        self_attention_mask = padding_mask(input_sen, input_sen).to(device)
        if seq_mask != None:
            self_attention_mask = torch.gt((self_attention_mask + seq_mask), 0)
        #print(vec_input)
        if self.use_vq:
            zq, mu, var, attentions, loss_vq, perplexity_vq, encoding_indices = self.vq_encoder(vec_input,self_attention_mask,var_factor)
        else:
            zq, attentions = self.encoder(vec_input,self_attention_mask)
            loss_vq = 0
            perplexity_vq = 0
            encoding_indices = 0
            mu = zq
            var = 0
        zq_splitted = self.splitter(zq)
        #classifier_output_list = []
        #for cid in range(len(classifier_list)):
        #    classifier_output_list.append(classifier_list[cid](zq_splitted[cid]))
        #print('zq_splitted[0]:',zq_splitted[0].shape)
        #print('zq_splitted[1]:',zq_splitted[1].shape)
        output_meter = self.classifier_meter(zq_splitted[0])
        output_grouping = self.classifier_grouping(zq_splitted[1])
        output_package = {'zq':zq,'attentions':attentions,'loss_vq':loss_vq,'perplexity_vq':perplexity_vq,
                          'encoding_indices':encoding_indices,'mu':mu,'var':var,
                          'output_meter':output_meter,'output_grouping':output_grouping}
        return output_package

    def MLM_predict(self,input_sen,input_token_ids,input_len,var_factor=0.02,seq_mask = None):
        vec_input = self.emb_layer(input_sen,input_token_ids,input_len)
        self_attention_mask = padding_mask(input_sen, input_sen)
        if seq_mask != None:
            self_attention_mask = torch.gt((self_attention_mask + seq_mask), 0)
        if self.use_vq:
            zq, mu, var, attentions, loss_vq, perplexity_vq, encoding_indices = self.vq_encoder(vec_input,self_attention_mask,var_factor)
        else:
            zq, attentions = self.encoder(vec_input,self_attention_mask)
            loss_vq = 0
            perplexity_vq = 0
            encoding_indices = 0
            mu = zq
            var = 0
        zq_splitted = self.splitter(zq)
        zq_detached = torch.cat((zq_splitted[0].detach(),zq_splitted[1].detach(),zq_splitted[2]),dim=2)
        output_pred = self.classifier_pred(zq_detached)
        output_package = {'zq':zq,'attentions':attentions,'loss_vq':loss_vq,'perplexity_vq':perplexity_vq,
                          'encoding_indices':encoding_indices,'mu':mu,'var':var,
                          'output_pred':output_pred}
        return output_package




