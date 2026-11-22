# Training assets and their licences

FENrir trains only on assets under these licences: CC0, CC BY, CC BY-SA,
MIT, Apache-2.0 and BSD. No GPL, AGPL, NonCommercial, NoDerivatives, or
assets with no licence. Add a row here for every new asset, with its source.

## Piece sets (`pieces/chess/`)

| Folder | Author | Source | Licence |
|---|---|---|---|
| `custom/a` to `custom/e` | Jost Triller | [Chess_diagram_to_FEN](https://github.com/tsoj/Chess_diagram_to_FEN) | CC0 1.0 |
| `extra/celtic`, `extra/eyes`, `extra/fantasy_alt`, `extra/spatial` | Maurizio Monge | [chess-art](https://github.com/maurimo/chess-art) | MIT |
| `svg/celtic`, `svg/fantasy`, `svg/spatial` | Maurizio Monge | [chess-art](https://github.com/maurimo/chess-art) | MIT |
| `svg/chessnut` | Alexis Luengas | [chessnut-pieces](https://github.com/LexLuengas/chessnut-pieces) | Apache-2.0 |
| `svg/rhosgfx` | RhosGFX | [rhosgfx.itch.io](https://rhosgfx.itch.io/) | CC0 1.0 |
| `svg/kiwen-suwi` | neverRare | [kiwen-suwi](https://github.com/neverRare) | CC BY 4.0 |
| `svg/firi` | James Faure | [Firi-pieceset](https://github.com/jfaure/Firi-pieceset) | CC BY 4.0 |
| `svg/cburnett` | Colin M.L. Burnett | [Wikimedia Commons](https://commons.wikimedia.org/wiki/Category:SVG_chess_pieces) (`Chess_*t45.svg`) | CC BY-SA 3.0 |
| `other/chess_1Kbyte_gambit`, `other/chess_kaneo`, `other/chess_kaneo_midnight`, `other/chess_maestro_bw` | Ka-hu | [chess-pieces](https://github.com/Ka-hu/chess-pieces) | CC BY 4.0 |

The Cburnett set is also in Lichess under GPLv2+. FENrir uses the Wikimedia
Commons copy under CC BY-SA 3.0.

## Board themes (`board_themes/chess/`)

| Folder | Author | Source | Licence |
|---|---|---|---|
| `custom/a.png`, `custom/b.png` | Jost Triller | [Chess_diagram_to_FEN](https://github.com/tsoj/Chess_diagram_to_FEN) | CC0 1.0 |
| `other/*.svg` | Ka-hu | [chess-pieces](https://github.com/Ka-hu/chess-pieces) | CC BY 4.0 |

The generator also draws plain and noisy boards in code.

## Other inputs

- **Fonts** for text noise: Google Fonts, downloaded at training time
  (mostly SIL OFL 1.1, some Apache-2.0). They are not committed.
- **Games** for positions: pychess-variants game records, downloaded by
  `download_pychess_games.sh`. Game moves are facts, not creative works.
- **Backgrounds** for the board finder: generated in code
  (`src/backgrounds.py`).
- **Pretrained backbones:** torchvision ImageNet weights. The code is
  BSD-3. The weights carry ImageNet's research terms.
