from typing import Dict, Mapping, Optional, Tuple, Any, Union, List, Dict, Optional, Tuple
import torch
from torch import nn, Tensor
from torch.distributions import Bernoulli
import torch.nn.functional as F
import torch.distributed as dist
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import pickle
import numpy as np
from torch_sparse import SparseTensor, matmul # Import matmul directly



class LearnableWeightedRGCN(nn.Module):
    def __init__(self, num_nodes, embedding_dim, graph_data_list, node_emb = None):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        
        # 1. PRE-TRANSFORM
        self.pre_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim), 
            nn.GELU()
        )

        # 2. PREPARE RELATIONS
        self.num_relations = len(graph_data_list)
        for i, (edge_index, edge_weight) in enumerate(graph_data_list):
            edge_index, edge_weight = gcn_norm(edge_index, edge_weight, num_nodes, add_self_loops=False)
            adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight,
                                 sparse_sizes=(num_nodes, num_nodes))
            rowptr, col, val = adj_t.csr()
            self.register_buffer(f'rowptr_{i}', rowptr)
            self.register_buffer(f'col_{i}', col)
            self.register_buffer(f'val_{i}', val)

        # 3. RELATION WEIGHTS
        self.relation_weights = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim, bias=False) 
            for _ in range(self.num_relations)
        ])
        
        # 4. SEMANTIC ATTENTION COMPONENTS (The Change)
        # We need a vector 'q' to score the messages.
        self.semantic_attn_vector = nn.Parameter(torch.empty(1, embedding_dim))
        nn.init.xavier_uniform_(self.semantic_attn_vector)
        self.semantic_act = nn.Tanh()

        # 5. REST OF MODEL
        self.self_loop_weight = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.bias = nn.Parameter(torch.zeros(embedding_dim))
        if node_emb is None:
            self.node_emb = nn.Embedding(num_nodes, embedding_dim)
        else:
            self.node_emb = node_emb
        nn.init.xavier_uniform_(self.node_emb.weight)
        self.final_norm = nn.LayerNorm(embedding_dim)

    def forward(self, target_node_indices=None):
        x_all = self.node_emb.weight
        x = self.pre_encoder(x_all)
        
        # List to store the output of each graph before aggregation
        messages_list = []
        
        # --- 1. COLLECT MESSAGES ---
        for i in range(self.num_relations):
            rowptr = getattr(self, f'rowptr_{i}')
            col = getattr(self, f'col_{i}')
            val = getattr(self, f'val_{i}')
            adj_t = SparseTensor(rowptr=rowptr, col=col, value=val, 
                                 sparse_sizes=(self.num_nodes, self.num_nodes),
                                 is_sorted=True)
            
            # W_r * h
            x_rel = self.relation_weights[i](x)
            
            # A_r * (W_r * h)
            msg = matmul(adj_t, x_rel)
            messages_list.append(msg)
            
        # Stack messages: Shape [Num_Nodes, Num_Relations, Emb_Dim]
        # Example: [1000, 12, 64]
        stacked_messages = torch.stack(messages_list, dim=1)
        
        # --- 2. CALCULATE ATTENTION ON MESSAGES ---
        # Transform messages to find their "importance features"
        # math: v = Tanh(h')
        
        
        # Calculate scores by dotting with the learnable attention vector
        attention_scores = (stacked_messages * self.semantic_attn_vector).sum(dim=2) # Shape [Num_Nodes, Num_Relations]
        
        # --- NEW CODE START ---
        # 1. Create a mask: True where the vector has data, False where it is all zeros.
        # Check along the feature dimension (dim=2).
        mask = (stacked_messages != 0).any(dim=2) # Shape [Num_Nodes, Num_Relations]

        # 2. Fill the scores corresponding to zero-vectors with negative infinity.
        # This ensures e^(-inf) = 0 in the softmax calculation.
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        # --- NEW CODE END ---

        # Normalize scores to get probability distribution
        attn_weights = F.softmax(attention_scores, dim=1) # Shape [Num_Nodes, Num_Relations]

        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        # --- 3. WEIGHTED AGGREGATION ---
        # Reshape weights to allow broadcasting: [Num_Nodes, Num_Relations, 1]
        alpha = attn_weights.unsqueeze(-1)
        
        # Sum( Weights * Messages )
        out_accum = (stacked_messages * alpha).sum(dim=1) # Shape [Num_Nodes, Emb_Dim]

        # --- 4. FINAL PROCESSING ---
        #self_loop = self.self_loop_weight(x)
        out = out_accum  + self.bias #+ self_loop
        out = F.gelu(out)
        
        control_emb = x_all[self.num_nodes-1].unsqueeze(0)
        final = torch.cat([out[:self.num_nodes-1], control_emb], dim=0)
        final = self.final_norm(final)
        
        if target_node_indices is not None:
            return final[target_node_indices], attn_weights, x_all
        return final, attn_weights, x_all


class LearnableMultiViewGNN_Sparse(nn.Module):
    def __init__(self, num_nodes, embedding_dim, graph_data_list, hops = 2, node_emb = None):
        super().__init__()
        self.num_nodes = num_nodes
        # 1. OPTIMIZED GRAPH SETUP
        self.hops = hops
        self.adj_t_list = []
        self.num_relations = len(graph_data_list)
        for i, (edge_index, edge_weight) in enumerate(graph_data_list):
            # Create Data object
            # Use GCNNorm explicitly. 
            # This calculates D^(-1/2) A D^(-1/2) to keep eigenvalues <= 1
            # We do this MANUALLY to ensure it works for your summed graph
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, num_nodes,
                add_self_loops=True # Adds self-loops automatically
            )
            
            # Convert to SparseTensor after normalization
            adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight,
                                 sparse_sizes=(num_nodes, num_nodes))
            
            rowptr, col, val = adj_t.csr()
            self.register_buffer(f'rowptr_{i}', rowptr, persistent=False)
            self.register_buffer(f'col_{i}', col, persistent=False)
            self.register_buffer(f'val_{i}', val, persistent=False)
        # 2. EMBEDDINGS
        # We need 20,000 graph nodes + 1 control node
        self.pre_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )
        if node_emb is None:
            self.node_emb = nn.Embedding(num_nodes, embedding_dim) 
        else:
            self.node_emb = node_emb
        nn.init.xavier_uniform_(self.node_emb.weight)
        self.final_norm = nn.LayerNorm(embedding_dim)
        # 3. VIEW ENCODERS (Replaces SGConv internals)
        # We just need a Linear layer and Norm for each view
        self.view_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU()
            ) for _ in range(len(graph_data_list))
        ])
        
        self.att_proj = nn.Linear(embedding_dim, 1) # Or embedding_dim for multi-head
        nn.init.constant_(self.att_proj.weight, 0.0)
        nn.init.constant_(self.att_proj.bias, 0.0)
        

    def forward(self, target_node_indices=None):
        x_all = self.node_emb.weight
        x_graph = self.pre_encoder(x_all)
        # Correct Slicing: Take the first 20,000 for the graph
        #x_graph = x_all
        
        embeddings = []
        for i in range(len(self.view_encoders)):
            

            # 1. Rebuild SparseTensor ON DEVICE
            # Since rowptr/col/val are buffers, they are already on the correct GPU
            rowptr = getattr(self, f'rowptr_{i}')
            col = getattr(self, f'col_{i}')
            val = getattr(self, f'val_{i}')
            
            # This reconstruction is extremely fast (zero copy)
            adj_t = SparseTensor(rowptr=rowptr, col=col, value=val, 
                                 sparse_sizes=(self.num_nodes, self.num_nodes),
                                 is_sorted=True)
            
            # 2. PROPAGATION (The "Conv")
            # We do K=2 hops manually using torch_sparse.matmul
            # This is safer than SGConv because it uses the installed torch_sparse kernel directly
            out = x_graph
            for _ in range(self.hops): # K=2
                out = matmul(adj_t, out) 
            
            # 2. RESIDUAL + ENCODER
            # Add residual connection for gradient flow!
            #out = out + x_graph 
            
            out = self.view_encoders[i](out)
            embeddings.append(out)
        # --- FUSION ---
        stack = torch.stack(embeddings, dim=1) 
        # (Insert your View Dropout Logic Here if Training)
        if len(self.view_encoders) > 1:
            scores = self.att_proj(stack) 
            alpha = F.softmax(scores, dim=1)
            fused = (alpha * stack).sum(dim=1)
        else:
            fused = stack.squeeze(1)
            alpha = None
        
        # Handle Control Token
        control_emb = x_all[self.num_nodes-1].unsqueeze(0)
        final = torch.cat([fused[:self.num_nodes-1], control_emb], dim=0)
        final = self.final_norm(final)
        if target_node_indices is not None:
            return final[target_node_indices], alpha, x_all
        return final, alpha, x_all





class PSDecoder(nn.Module):
    """
    Decoder for ps score prediction.
    revised from scGPT.ClsDecoder
    """

    def __init__(
        self,
        d_model: int,
        n_pert: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
        geneinput: bool = False,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        if geneinput:
            self.input_dim =  d_model * 2 #this is a concatenation of cell embedding and perturbation embedding
        else:
            self.input_dim = d_model # just cell embedding
        
        for i in range(nlayers - 1):
            if i == 0:
                self._decoder.append(nn.Linear(self.input_dim, d_model))
            else:
                self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_pert)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)

class Batch2LabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return x



class PertExpAE(nn.Module):
    """
    Concatenating gene expression embeddings (from transformers) with perturbation embeddings (from scGPT's PertEncoder)
    """
    def __init__(
        self,
        d_model: int,
        d_hid: int
    ):
        super().__init__()
        d_in = d_model
        #d_in = d_model
        self.d_in = d_in
        self.d_hid = d_hid
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hid * 2),
            nn.LeakyReLU(),
            nn.Linear(d_hid * 2, d_hid),
            nn.LeakyReLU(),
            nn.LayerNorm(d_hid)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_hid, d_hid * 2),
            nn.LeakyReLU(),
            nn.Linear(d_hid * 2, d_in),
        )


    def forward(self, z: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer concatenated with perturbation embedding, (batch, d_model*2)"""
        # pred_value = self.fc(x).squeeze(-1)  
        x, y = torch.split(z, [self.d_in, self.d_hid], dim=1)
        encoded = self.encoder(x)+y
        decoded = self.decoder(encoded)
        return decoded # (batch, d_model)


class PSDecoder(nn.Module):
    """
    Decoder for ps score prediction.
    revised from scGPT.ClsDecoder
    """

    def __init__(
        self,
        d_model: int,
        n_pert: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(nn.LayerNorm(d_model))
            self._decoder.append(activation())
            
        self.out_layer = nn.Linear(d_model, n_pert)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)

class Batch2LabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return x

class BatchLabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return x


class PertLabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)
        print(self)
    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return x

    
class PertExpEncoder(nn.Module):
    """
    Concatenating gene expression embeddings (from transformers) with perturbation embeddings (from scGPT's PertEncoder)
    """
    def __init__(
        self,
        d_model: int
    ):
        super().__init__()
        d_in = d_model * 2 
        #d_in = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            #nn.Sigmoid(),
            #nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            #nn.ReLU(),
            #nn.Sigmoid(),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            #nn.Linear(d_model, d_model),
        )

        print(self)
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer concatenated with perturbation embedding, (batch, d_model*2)"""
        # pred_value = self.fc(x).squeeze(-1)  
        return self.fc(x) # (batch, d_model)
    


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # TODO: test using actual embedding layer if input is categorical
        # expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class CategoryValueEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

# added here for potential customisations
class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
        expr_activation: str = 'linear'
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(4*d_model, 1),
            
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            )
        self.expr_activation = expr_activation
        self.expr_act = ExpressionActivate(activation = self.expr_activation)
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.expr_act(self.fc(x).squeeze(-1))  # (batch, seq_len)
        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x).squeeze(-1)  # (batch, seq_len)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)
        # TODO: note that the return currently is only for training. Since decoder
        # is not used in the test setting for the integration task, the eval/inference
        # logic is not implemented yet. However, remember to implement it when
        # the decoder is used in any test setting. The inference logic will need
        # to sample from the bernoulli distribution with the zero_probs.

# added here for potential customisations
class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(nn.LayerNorm(d_model)) # switched the normalization to before activation
            self._decoder.append(activation())
            
        self.out_layer = nn.Linear(d_model, n_cls)
        print(self)
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)

class ExpressionActivate(nn.Module):
    def __init__(self, activation='elu', elu_alpha = 1):
        super().__init__()
        self.activation = activation
        self.elu_alpha = elu_alpha
        if activation == 'elu':
            self.pred_act = F.elu
        elif activation == 'relu':
            self.pred_act = F.relu
        elif activation == 'exponential':
            self.pred_act = torch.exp
        elif activation == 'softplus':
            self.pred_act = F.softplus


    def forward(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'softmax':
            if x.shape[-1] <= 1:
                raise ValueError("Input vector length M must be greater than 1 "
                             "to split into energy and distribution parts.")

            # 1. Split the vector into the energy logit (first element) and
            #    the distribution logits (the rest).
            # energy_logit shape: (batch_size, 1)
            # distribution_logits shape: (batch_size, M-1)
            energy_logit, distribution_logits = torch.split(x, [1, x.shape[-1] - 1], dim=-1)

            # 2. Apply a non-negative activation to the energy logit.
            # Softplus ensures the magnitude component is always positive.
            activated_energy = F.softplus(energy_logit)

            # 3. Apply softmax to the rest of the vector to get a probability distribution.
            distribution = F.softmax(distribution_logits, dim=-1) 

            output = activated_energy * distribution

            # 4. Concatenate the activated energy and the distribution to form the final vector.
            output = torch.cat([activated_energy, output], dim=-1)

            return output
        if self.activation == 'square':
            return torch.square(x)
        
        return self.pred_act(x)+self.elu_alpha if self.activation == 'elu' else self.pred_act(x)


class MVCDecoder(nn.Module):
    """
    Decoder for the masked value prediction for cell embeddings.
    """

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
        expr_activation: str = 'linear'
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super().__init__()
        self.expr_activation = expr_activation
        d_in = d_model * 2 if use_batch_labels else d_model
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = nn.LeakyReLU()#query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:  # by default, gene-wise prob rate
                self.W_zero_logit = nn.Linear(d_model, d_in)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")
        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob
        self.expr_act = ExpressionActivate(activation = self.expr_activation)
    def forward(
        self, cell_emb: Tensor, gene_embs: Tensor
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = self.query_activation(cell_emb.unsqueeze(2))  # (batch, embsize, 1)
            # the pred gene expr values, # (batch, seq_len)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            pred_value = self.expr_act(pred_value)
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            # zero logits need to based on the cell_emb, because of input exprs
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            # expand cell_emb to (batch, seq_len, embsize)
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)

            h = self.hidden_activation(
                self.fc1(torch.cat([cell_emb, query_vecs], dim=2))
            )
            if self.explicit_zero_prob:
                raise NotImplementedError
            pred_value = self.fc2(h).squeeze(2)
            pred_value = self.expr_act(pred_value)
            return pred_value  # (batch, seq_len)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            pred_value = self.fc2(h).squeeze(2)
            pred_value = self.expr_act(pred_value)
            return pred_value  # (batch, seq_len)


class AutoDiscretizationEmbedding(nn.Module):
    def __init__(self, dim, bin_num, bin_alpha, mask_token_id = None, pad_token_id = None):
        super().__init__()
        
        self.dim = dim
        self.bin_num = bin_num
        self.bin_alpha = bin_alpha
        
        self.mlp = nn.Linear(1, self.bin_num)
        self.mlp2 = nn.Linear(self.bin_num, self.bin_num)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.Softmax = nn.Softmax(dim=-1)
        self.emb = nn.Embedding(self.bin_num, self.dim)
        
        self.emb_mask = nn.Embedding(1, self.dim)
        self.emb_pad = nn.Embedding(1, self.dim)
        
        self.bin_num_idx = torch.tensor(range(self.bin_num))
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

        self.tensor0 = torch.tensor(0, dtype=torch.long)

    def forward(self, x, output_weight=0):
        x_mask_idx = (x==self.mask_token_id).nonzero()
        x_pad_idx = (x==self.pad_token_id).nonzero()
        
        x = self.mlp(x) # [B,N,1] -> [B,N,H]
        x = self.LeakyReLU(x) # [B,N,H]
        x_crosslayer = self.mlp2(x) # [B,N,H]
        x = self.bin_alpha * x + x_crosslayer # [B,N,H]
        weight = self.Softmax(x) # [B, N, H]
        
        bin_num_idx = self.bin_num_idx.to(x.device) # [H,]
        
        token_emb = self.emb(bin_num_idx) # [H, D]
        x = torch.matmul(weight, token_emb) #[B, N, D]

        tensor0 = torch.tensor(0, dtype=torch.long, device=x.device)

        mask_token_emb = self.emb_mask(tensor0).to(x.device).type(x.dtype)
        x[x_mask_idx[:,0],x_mask_idx[:,1],:] = mask_token_emb.repeat(x_mask_idx.shape[0],1)

        pad_token_emb = self.emb_pad(tensor0).to(x.device).type(x.dtype)
        x[x_pad_idx[:,0],x_pad_idx[:,1],:] = pad_token_emb.repeat(x_pad_idx.shape[0],1)
    
        if output_weight:
            return x,weight
        return x
        

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
    
from torch.autograd import Function
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)

class AdversarialDiscriminator(nn.Module):
    """
    Discriminator for the adversarial training for batch correction.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = False,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)
    
class PerturbationDecoder(nn.Module):
    """
    Decoder for perturbation label prediction.
    revised from scGPT.ClsDecoder
    """

    def __init__(
        self,
        d_model: int,
        n_pert: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_pert)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)




class GenePTHybridEmbedding(nn.Module):
    """
    A memory-efficient embedding layer that selectively loads GenePT embeddings.
    It supports hybrid training: keeping known genes frozen while training unknown ones,
    and switching to full fine-tuning later.
    """
    def __init__(self, 
                 genept_pickle_path: str, 
                 vocab: Dict, 
                 embedding_dim: int = 3072,
                 padding_idx = None
                 ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        
        # 1. Create the unified embedding layer
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=padding_idx)
        
        # 2. Load and Filter Data (Memory Efficiency Step)
        self._init_weights_from_pickle(genept_pickle_path)
        
        # 3. Setup Gradient Masking for Semi-Trainable functionality
        # We create a mask: 1.0 for trainable (unknown genes), 0.0 for frozen (known genes)
        # This mask is registered as a buffer so it saves with the model state_dict
        self.register_buffer('grad_mask', torch.ones(self.vocab_size, 1))
        
        # Identify which indices correspond to known GenePT genes to create the mask
        self._setup_gradient_mask()

        # State flag
        self.pretrained_frozen = False
        self._hook_handle = None

    def _init_weights_from_pickle(self, path: str):
        """
        Loads the pickle, filters for only target genes, and initializes weights.
        """
        print(f"Loading GenePT data from {path}...")
        try:
            with open(path, 'rb') as f:
                full_genept_dict = pickle.load(f)
        except FileNotFoundError:
            # Fallback for demonstration if file doesn't exist
            print("Warning: GenePT file not found. Initializing randomly for demo.")
            full_genept_dict = {}

        # Tracking stats
        self.found_genes_indices = []
        self.missing_genes_indices = []
        
        # Temporary tensor to hold new weights
        with torch.no_grad():
            # Initialize everything with Xavier Uniform first (good for the missing genes)
            nn.init.xavier_uniform_(self.embedding.weight)
            
            # Overwrite with GenePT data where matches exist
            for gene, idx in self.vocab.stoi.items():
                if gene in full_genept_dict:
                    # Retrieve and cast to tensor
                    emb_vec = torch.tensor(full_genept_dict[gene], dtype=torch.float32)
                    
                    # Safety check on dimension
                    if len(emb_vec) != self.embedding_dim:
                        raise ValueError(f"GenePT dim {len(emb_vec)} != expected {self.embedding_dim}")
                        
                    self.embedding.weight[idx] = emb_vec
                    self.found_genes_indices.append(idx)
                else:
                    self.missing_genes_indices.append(idx)
        
        print(f"Initialized: {len(self.found_genes_indices)} matches found, "
              f"{len(self.missing_genes_indices)} initialized randomly.")

    def _setup_gradient_mask(self):
        """
        Sets up the binary mask: 0 for known genes (frozen), 1 for unknown (trainable).
        """
        # Default is 1 (trainable)
        self.grad_mask.fill_(1.0)
        
        # Set known genes to 0 (frozen)
        if self.found_genes_indices:
            self.grad_mask[self.found_genes_indices] = 0.0

    def _backward_hook(self, grad):
        """
        The magic hook: Multiplies gradients by the mask.
        If mask is 0, gradient becomes 0 -> No weight update.
        """
        # Expand mask to match gradient shape (vocab_size, emb_dim)
        return grad * self.grad_mask

    def freeze_pretrained(self):
        """
        Locks the weights of genes found in GenePT. 
        Genes NOT found in GenePT remain trainable.
        """
        if self.pretrained_frozen:
            return # Already frozen
            
        # Register the backward hook on the embedding WEIGHTS specifically
        self._hook_handle = self.embedding.weight.register_hook(self._backward_hook)
        self.pretrained_frozen = True
        print("Status: GenePT embeddings FROZEN. Unmatched genes remain TRAINABLE.")

    def unfreeze_all(self):
        """
        Removes the gradient mask. All genes (including GenePT matches) 
        become fully trainable.
        """
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        
        self.pretrained_frozen = False
        print("Status: All embeddings UNFROZEN (Fully Trainable).")

    def apply_freeze_mask(self, embeddings: torch.Tensor, indices: torch.Tensor):
        """
        Applies the freeze mask to a specific branch of the data flow.
        
        Args:
            embeddings: Tensor (Batch, Seq, Dim)
            indices: Tensor (Batch, Seq) - needed to look up the mask
            
        Returns:
            masked_embeddings: Same content as input, but gradients are filtered.
        """
        # 1. Look up the mask for these specific genes
        # mask shape: (Batch, Seq, 1)
        current_mask = F.embedding(indices, self.grad_mask)
        
        # 2. Define the Hook Function
        def hook_fn(grad):
            # Multiply incoming gradient by the mask
            # If mask is 0 (frozen), grad becomes 0.
            return grad * current_mask
            
        # 3. Register Hook on this specific tensor instance
        # We must clone or create a view to attach a hook safely in the graph
        embeddings_hooked = embeddings.view_as(embeddings)
        embeddings_hooked.register_hook(hook_fn)
        
        return embeddings_hooked

    def forward(self, gene_indices):
        return self.embedding(gene_indices)


class GenePTHead(nn.Module):
    """
    The Comprehensive Module.
    Wraps the embeddings and the two task heads (Transformer & GNN).
    """
    def __init__(self, 
                 raw_emb,
                 emb_dim: int = 512
                 ):
        super().__init__()
        
        # Input dim is fixed by GenePT
        input_dim = 3072
        
        # 2. Task Head 1: Transformer Projection
        self.head = nn.Sequential(
            #nn.Linear(input_dim, input_dim // 2),
            #nn.ReLU(),
            nn.Linear(input_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )
        self.raw_emb = raw_emb
    def forward(self, gene_indices: torch.Tensor):
        """
        Args:
            gene_indices: Tensor of shape (Batch, Seq_Len)
            task: 'transformer', 'gnn', or 'both'
        """
        # 1. Get Base Embeddings (Hook point A)
        raw_embeddings = self.raw_emb(gene_indices)
        out = self.head(raw_embeddings)
        return out


class CrossAttn(nn.Module):
    """
    Expands the limited Transformer context to the full Gene Vocabulary 
    using Cross Attention.
    TODO This needs to be reimplemented with flash attention and Padding
    """
    def __init__(self, 
                 d_model: int,  
                 nhead: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        # 2. Cross Attention
        # Query = Static Global Genes
        # Key/Value = Dynamic Transformer Output
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            batch_first=True,
            dropout=dropout
        )
        
        # 3. Output Norms
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor, 
                value: torch.Tensor
                ):
        """
        Args:
            transformer_context: (Batch, Seq_Len, d_model) 
                                 The output of the transformer (active genes).
            raw_vocab_embeddings: (Vocab_Size, genept_dim)
                                  The full matrix from GenePTHybridEmbedding.
        
        Returns:
            expanded_embeddings: (Batch, Vocab_Size, d_model)
        """
  
        # B. Run Cross Attention
        # Q: All Genes, K: Context Genes, V: Context Genes
        attn_out, _ = self.cross_attn(
            query=query,
            key=key,
            value=value
        )
        
        # C. Residual Connection + Norm
        # We add the projected_queries (static) to the attn_out (dynamic context).
        # This ensures genes with no relevant context still have a base representation.
        expanded_embeddings = self.norm(query + self.dropout(attn_out))
        
        return expanded_embeddings