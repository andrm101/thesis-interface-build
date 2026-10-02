<img src="https://raw.githubusercontent.com/andrm101/thesis-interface-build/main/docs/wiki/images/cover-web-dashboard.svg" alt="Web Dashboard" width="100%">

The web dashboard puts the main analyses in a browser. It has two parts:
- **Front end:** an Angular app, in `web/`.
- **Back end:** a FastAPI service, in `api/`, that calls the same Python functions as `reproduce.py`. Its numbers therefore match [`results/RESULTS.md`](https://github.com/andrm101/thesis-interface-build/blob/main/results/RESULTS.md).

![Overview page](https://raw.githubusercontent.com/andrm101/thesis-interface-build/main/docs/img/dashboard-overview.png)

## Pages

| Page | What you can do |
|---|---|
| **Overview** | See a map of Europe coloured by R&D cluster or convergence club, the R&D typology (PCA of the K-Means clusters), the Phillips–Sul transition paths and the key findings, each linked to its result section. |
| **Explore** | Plot any variable over time, highlight up to six countries against the rest, map any year or press *Play* to animate 2000–2023 (click a country to highlight it), and rank countries in any year. |
| **Local projections** | Run impulse responses live, in % or in levels, for all countries or split Innovative / Emerging. Presets reproduce D4, D5 and B8. |
| **Event study** | Run Callaway–Sant'Anna on the EU accession dates, the R&D tax-reform dates or your own `Country:Year` list. Choose never-treated or not-yet-treated controls, optional detrending, and 49–499 bootstrap draws. |
| **Data coverage** | See which country-variable pairs have gaps over 2000–2023. |
| **Results & figures** | Browse every section of `RESULTS.md`, and the thesis figures as PNG and PDF. |

![Local projections, dark theme](https://raw.githubusercontent.com/andrm101/thesis-interface-build/main/docs/img/dashboard-projections.png)

## Running it

**Docker.** This works on any machine with Docker and needs nothing else:

```bash
git clone https://github.com/andrm101/thesis-interface-build.git
cd thesis-interface-build
docker compose up --build        # → http://localhost:8000
```

**Without Docker.** You need Python ≥ 3.10 and Node ≥ 22.22.3 (or 24 LTS):

```bash
pip install -r requirements-api.txt
cd web && npm ci && npm run build && cd ..
uvicorn api.main:app --port 8000
```

**Developing the front end.**
- Run `uvicorn api.main:app --reload` in one terminal and `npm start` in `web/` in another.
- Then open http://localhost:4200. The page reloads on save, and API calls are forwarded to port 8000.
- The API's interactive documentation is at http://localhost:8000/docs.

## How it is built

| Layer | Choice | Why |
|---|---|---|
| Back end | FastAPI + Pydantic | Wraps `causal.py`, `clubs.py` and `reproduce.py` with typed requests. The panel loads once at start-up. |
| Front end | Angular 22: standalone components, signals, lazily loaded pages | Each page downloads only when it is first opened. |
| Charts | Apache ECharts via ngx-echarts | Built-in tooltips, legends, heatmaps and maps. |
| Maps | Natural Earth 1:50m outlines, Lambert azimuthal equal-area projection | Country areas stay comparable, as in Eurostat's own maps. Variables that cross zero, such as the fiscal balance, use a diverging scale centred on 0. Countries outside the panel are hatched. `web/scripts/build-europe-map.mjs` rebuilds the outline file. |
| Colours | The validated categorical palette of `figures.py` | The web charts and the thesis PDFs match, and both themes pass colour-blind checks. |
| Deployment | One Docker image | FastAPI serves the built Angular app and the API from the same address. |

**Speed.** Local projections return in under a second. The event study scales with the number of bootstrap draws: about 4 s with 49 draws and 9 s with 99.

<img src="https://raw.githubusercontent.com/andrm101/thesis-interface-build/main/assets/brand-divider.svg" alt="" width="100%">
