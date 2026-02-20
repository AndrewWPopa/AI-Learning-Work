    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TransformerModel(nn.Module):
        def __init__(self, vocab_size, embed_size, block_size, num_heads, num_layers):
            super().__init__()

            self.token_embedding = nn.Embedding(vocab_size, embed_size)
            self.position_embedding = nn.Embedding(block_size, embed_size)

            self.blocks = nn.ModuleList([
                TransformerBlock(embed_size, num_heads)
                for _ in range(num_layers)
            ])

            self.ln_f = nn.LayerNorm(embed_size)
            self.head = nn.Linear(embed_size, vocab_size)

        def forward(self, x, targets=None):
            B, T = x.shape

            token_emb = self.token_embedding(x)
            positions = torch.arange(T, device=x.device)
            pos_emb = self.position_embedding(positions)

            x = token_emb + pos_emb

            for block in self.blocks:
                x = block(x)

            x = self.ln_f(x)
            logits = self.head(x)

            loss = None
            if targets is not None:
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets)

            return logits, loss

    class TransformerBlock(nn.Module):
        def __init__(self, embed_size, num_heads):
            super().__init__()

            self.attn = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)
            self.ln1 = nn.LayerNorm(embed_size)

            self.ff = nn.Sequential(
                nn.Linear(embed_size, 4 * embed_size),
                nn.ReLU(),
                nn.Linear(4 * embed_size, embed_size),
            )

            self.ln2 = nn.LayerNorm(embed_size)

        def forward(self, x):
            attn_output, _ = self.attn(x, x, x)
            x = self.ln1(x + attn_output)

            ff_output = self.ff(x)
            x = self.ln2(x + ff_output)

            return x