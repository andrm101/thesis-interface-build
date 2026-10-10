# Explainer video

An 80-second explainer of the thesis, drawn by code. Each frame is a function of time rendered from real results, the same technique as Claude Motion. Outputs are in `docs/media/`:
- `thesis-explainer.mp4`: 1280×720, H.264;
- `thesis-explainer-map.gif`: a preview of the map scene;
- `thesis-explainer-poster.png`.

```bash
python assets/motion/export_data.py      # numbers → data.json (same functions as reproduce.py)
npm i playwright                          # once
node assets/motion/render.mjs --fps 30    # frames → MP4 + GIF (needs ffmpeg, or set FFMPEG=…)
```

| File | Role |
|---|---|
| `export_data.py` | Computes everything the video shows: R&D intensity by country and year, typology, convergence clubs, the Granger test (B6), and the D2, D4 and D5 estimates. It also projects the Europe outlines (Lambert equal-area). |
| `explainer.html` | `render(t)` draws the frame at time *t*. Opened in a browser, the page plays in real time. |
| `render.mjs` | Captures every frame with Playwright and encodes it with ffmpeg. |

## Scenes

| Seconds | Scene |
|---|---|
| 0–6 | Title |
| 6–14 | The GERD paradox |
| 14–25 | Two R&D regimes, on the map |
| 25–42 | R&D intensity, 2000→2023 |
| 42–52 | Which way causality runs |
| 52–63 | Convergence clubs |
| 63–75 | Policy levers: tax reforms, public R&D budgets, fiscal tightening |
| 75–80 | Outro |

**Rebuilding.** Rerun both commands after `reproduce.py` changes, so the video stays in step with the results.

**Colours.** The colours are the validated palette, checked against the navy background.
