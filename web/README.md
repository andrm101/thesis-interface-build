# EU Innovation Panel — web dashboard

Angular 22 front end (standalone components, signals, ngx-echarts) for the
FastAPI back end in `../api`. See the main README → *Web dashboard* for how
to run it.

```bash
npm ci
npm start        # dev server on :4200, proxies /api and /figures to :8000
npm run build    # production build → dist/web/browser (served by FastAPI)
```

Layout:

- `src/app/core/` — API client and response types, theme (light/dark tokens),
  shared ECharts chrome
- `src/app/pages/` — one lazily loaded component per page
- `public/europe.json` — Europe outlines (Natural Earth 1:50m), rebuilt with
  `node scripts/build-europe-map.mjs`
- `src/styles.css` — design tokens; chart colours are the validated palette
  used by `figures.py`
