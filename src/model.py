import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import config


class CharCNN(nn.Module):
    
    def __init__(self, char_vocab_size: int, char_embedding_dim: int, filters: int, kernel_sizes: List[int]):
        super().__init__()
        
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0)
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(char_embedding_dim, filters, kernel_size=k, padding=k-1)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(config.CHAR_DROPOUT_RATE)
        
    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, max_word_len = char_ids.size()
        
        char_ids_flat = char_ids.view(-1, max_word_len) 
        
        char_emb = self.char_embedding(char_ids_flat)  
        char_emb = self.dropout(char_emb)
        
        char_emb = char_emb.transpose(1, 2)  

        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = F.relu(conv(char_emb))  
            pool_out = torch.max(conv_out, dim=2)[0]  
            conv_outputs.append(pool_out)
        
        char_vecs = torch.cat(conv_outputs, dim=1)  
        
        char_vecs = char_vecs.view(batch_size, seq_len, -1)
        
        return char_vecs


class MultiHeadAttention(nn.Module):
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        
    def forward(self, values: torch.Tensor, keys: torch.Tensor, query: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.shape[0]
        
        Q = self.query(query)
        K = self.key(keys)
        V = self.value(values)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e10"))
        
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.hidden_dim)
        
        out = self.fc_out(context)
        return out


class CRFLayer(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags

        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(self, emissions, tags, mask):
        log_likelihood = self._log_likelihood(emissions, tags, mask)
        return -log_likelihood.mean()

    def _log_likelihood(self, emissions, tags, mask):
        numerator = self._score_sentence(emissions, tags, mask)
        denominator = self._compute_partition(emissions, mask)
        return numerator - denominator

    def _score_sentence(self, emissions, tags, mask):
        batch_size, seq_len, _ = emissions.shape

        score = self.start_transitions[tags[:, 0]]
        score += emissions[torch.arange(batch_size), 0, tags[:, 0]]

        for t in range(1, seq_len):
            transition_score = self.transitions[
                tags[:, t - 1], tags[:, t]
            ]
            emission_score = emissions[torch.arange(batch_size), t, tags[:, t]]

            score += (transition_score + emission_score) * mask[:, t]

        seq_ends = mask.sum(1).long() - 1
        last_tags = tags.gather(1, seq_ends.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]

        return score

    def _compute_partition(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape

        alpha = self.start_transitions + emissions[:, 0]

        for t in range(1, seq_len):
            emission = emissions[:, t].unsqueeze(2)
            transition = self.transitions.unsqueeze(0)
            alpha_exp = alpha.unsqueeze(1)

            scores = alpha_exp + transition + emission
            new_alpha = torch.logsumexp(scores, dim=2)

            alpha = torch.where(
                mask[:, t].unsqueeze(1).bool(),
                new_alpha,
                alpha
            )

        alpha += self.end_transitions
        return torch.logsumexp(alpha, dim=1)

    def decode(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape

        viterbi = self.start_transitions + emissions[:, 0]
        backpointers = []

        for t in range(1, seq_len):
            broadcast_viterbi = viterbi.unsqueeze(2)
            broadcast_trans = self.transitions.unsqueeze(0)

            scores = broadcast_viterbi + broadcast_trans
            best_scores, best_paths = scores.max(dim=1)

            viterbi = best_scores + emissions[:, t]
            backpointers.append(best_paths)

        viterbi += self.end_transitions
        best_last_tags = viterbi.argmax(dim=1)

        best_paths = []

        for b in range(batch_size):
            seq_len_b = int(mask[b].sum().item())
            best_tag = int(best_last_tags[b].item())
            path = [best_tag]

            for backpointer in reversed(backpointers[:seq_len_b - 1]):
                best_tag = int(backpointer[b][best_tag].item())
                path.append(best_tag)

            best_paths.append(path[::-1])

        return best_paths    

    
class SOTABiLSTMPOSTagger(nn.Module):    
    def __init__(self, word_vocab_size: int, char_vocab_size: int, num_tags: int, 
                 embedding_dim: int = config.EMBEDDING_DIM, 
                 hidden_dim: int = config.HIDDEN_DIM):
        super().__init__()
        
        self.num_tags = num_tags
        self.hidden_dim = hidden_dim
        
        self.char_cnn = CharCNN(
            char_vocab_size, 
            config.CHAR_EMBEDDING_DIM,
            config.CHAR_NUM_FILTERS,
            config.CHAR_KERNEL_SIZES
        )
        char_cnn_output_dim = len(config.CHAR_KERNEL_SIZES) * config.CHAR_NUM_FILTERS
        
        self.word_embedding = nn.Embedding(word_vocab_size, embedding_dim, padding_idx=0)
        
        embedding_total_dim = embedding_dim + char_cnn_output_dim
        if config.LAYER_NORM:
            self.embedding_norm = nn.LayerNorm(embedding_total_dim)
        
        self.lstm = nn.LSTM(
            embedding_total_dim,
            hidden_dim,
            num_layers=config.NUM_LSTM_LAYERS,
            batch_first=True,
            bidirectional=config.BIDIRECTIONAL,
            dropout=config.DROPOUT_RATE if config.NUM_LSTM_LAYERS > 1 else 0
        )
        
        lstm_output_dim = hidden_dim * 2 if config.BIDIRECTIONAL else hidden_dim
        
        self.attention = None
        if config.USE_ATTENTION:
            self.attention = MultiHeadAttention(lstm_output_dim, num_heads=4)
        
        self.crf = None
        if config.USE_CRF:
            self.crf = CRFLayer(num_tags)
        
        self.output_layer = nn.Linear(lstm_output_dim, num_tags)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        
    def forward(self, word_ids: torch.Tensor, char_ids: Optional[torch.Tensor] = None, 
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = word_ids.size()
        
        word_emb = self.word_embedding(word_ids)
        
        if char_ids is not None:
            char_emb = self.char_cnn(char_ids)
            emb = torch.cat([word_emb, char_emb], dim=-1)
        else:
            emb = word_emb
        
        if config.LAYER_NORM:
            emb = self.embedding_norm(emb)
        
        emb = self.dropout(emb)
        
        if lengths is not None:
            emb = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(emb)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(emb)
        
        if self.attention is not None:
            lstm_out = self.attention(lstm_out, lstm_out, lstm_out)
        
        lstm_out = self.dropout(lstm_out)
        
        logits = self.output_layer(lstm_out)
        
        return logits


def create_model(word_vocab_size: int, char_vocab_size: int, num_tags: int, device: torch.device) -> SOTABiLSTMPOSTagger:
    model = SOTABiLSTMPOSTagger(word_vocab_size, char_vocab_size, num_tags)
    return model.to(device)