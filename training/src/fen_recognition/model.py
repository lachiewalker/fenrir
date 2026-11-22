import torch
import torch.nn as nn
import torchvision.models as models

from src import consts
from src.games import get_game

TILE_EMBED_DIM = 512
FULL_EMBED_DIM = 512


BACKBONES = ("convnext_tiny", "mobilenet_v3_large")


def _embedding_model(backbone: str, embed_dim: int) -> nn.Module:
    if backbone == "convnext_tiny":
        result = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        features = 768
    elif backbone == "mobilenet_v3_large":
        result = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        features = 960
    else:
        raise ValueError(f"Unknown backbone {backbone}, expected one of {BACKBONES}")
    result.classifier = nn.Sequential(
        torch.nn.Flatten(start_dim=1, end_dim=-1),
        torch.nn.Linear(in_features=features, out_features=embed_dim),
        nn.ReLU(),
    )
    return result


class BoardRec(nn.Module):
    def __init__(
        self,
        game: str,
        tile_size: int = consts.DEFAULT_TILE_SIZE,
        dropout: float = 0.1,
        backbone: str = "mobilenet_v3_large",
    ):
        super().__init__()
        self.game = get_game(game)
        self.tile_size = tile_size
        self.board_h, self.board_w = consts.board_pixel_size(self.game, tile_size)
        self.num_squares = self.game.num_squares
        self.out_channels = len(self.game.piece_symbols) + 1

        self.backbone = backbone
        self.tile = _embedding_model(backbone, TILE_EMBED_DIM)
        self.full = _embedding_model(backbone, FULL_EMBED_DIM)

        # Positional embeddings added to tile features before concatenation
        self.tile_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_squares, TILE_EMBED_DIM)
        )
        nn.init.trunc_normal_(self.tile_pos_embed, std=0.02)

        # The projection of [tile, board] is split in two, so the board term is
        # broadcast instead of expanded and concatenated. tract's optimiser
        # fails on the expand-concat-matmul form.
        self.embed_dim = 512
        self.tile_proj = nn.Linear(TILE_EMBED_DIM, self.embed_dim)
        self.full_proj = nn.Linear(FULL_EMBED_DIM, self.embed_dim, bias=False)

        # Transformer block: Attention → Add&Norm → FFN → Add&Norm
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(self.embed_dim)

        # Classification head
        self.dense = nn.Sequential(
            nn.Linear(self.embed_dim, 768),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.out_channels),
        )

    def forward(self, img):
        batch_size, ch, h, w = img.shape

        assert h == self.board_h
        assert w == self.board_w
        assert ch == 3

        rows, cols, ts = self.game.board_rows, self.game.board_cols, self.tile_size
        x = img.reshape(batch_size, ch, rows, ts, cols, ts)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(batch_size * rows * cols, ch, ts, ts)

        x = self.tile(x)
        x = x.reshape(batch_size, self.num_squares, -1)
        x = x + self.tile_pos_embed

        z = self.full(img)
        x = torch.relu(self.tile_proj(x) + self.full_proj(z).unsqueeze(1))

        # Transformer block
        attn_out, _ = self.attention(x, x, x)
        x = self.attn_norm(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)

        # Per-square classification
        x = self.dense(x)

        return x



def upgrade_state_dict(state: dict) -> dict:
    """Maps an upstream checkpoint, with one pre_attn_dense layer, to the split projection."""
    if "pre_attn_dense.0.weight" not in state:
        return state
    state = dict(state)
    weight = state.pop("pre_attn_dense.0.weight")
    state["tile_proj.weight"] = weight[:, :TILE_EMBED_DIM]
    state["full_proj.weight"] = weight[:, TILE_EMBED_DIM:]
    state["tile_proj.bias"] = state.pop("pre_attn_dense.0.bias")
    return state
