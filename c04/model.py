"""BiLSTM sentence classifier with masked attention or final-state pooling."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, dropout=0.3,
                 classes=150, pooling="attention", embedding_trainable=True):
        super().__init__()
        if pooling not in {"attention", "last"}:
            raise ValueError("pooling must be attention or last")
        self.pooling = pooling
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix.clone(), freeze=not embedding_trainable, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_dim,
                            batch_first=True, bidirectional=True)
        self.attention = nn.Linear(2*hidden_dim, 1, bias=False) if pooling == "attention" else None
        self.classifier = nn.Linear(2*hidden_dim, classes)

    def forward(self, tokens, lengths):
        embedded = self.dropout(self.embedding(tokens))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(packed)
        if self.pooling == "attention":
            output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=tokens.shape[1])
            mask = torch.arange(tokens.shape[1], device=tokens.device)[None, :] < lengths.to(tokens.device)[:, None]
            attention = self.attention(output).squeeze(-1).masked_fill(~mask, float("-inf"))
            weights = torch.softmax(attention, dim=1)
            sentence = (output*weights.unsqueeze(-1)).sum(dim=1)
        else:
            # The final forward/backward states exclude padding and respect true lengths.
            sentence = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(self.dropout(sentence))
