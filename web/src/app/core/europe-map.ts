import { HttpClient } from '@angular/common/http';
import { Component, Injectable, computed, inject, input, output } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import type { EChartsOption } from 'echarts';
import { NgxEchartsDirective } from 'ngx-echarts';
import { from, map, shareReplay, switchMap } from 'rxjs';
import { base, fmt } from './charts';
import { ThemeService } from './theme.service';

/** Loads public/europe.json (Natural Earth 1:50m, built by
 *  scripts/build-europe-map.mjs) and registers it with ECharts once. */
@Injectable({ providedIn: 'root' })
export class EuropeMapLoader {
  private http = inject(HttpClient);
  readonly ready$ = this.http.get<object>('europe.json').pipe(
    switchMap(geo => from(import('echarts')).pipe(map(ec => {
      ec.registerMap('europe', geo as never);
      return true;
    }))),
    shareReplay(1));
}

// Lambert azimuthal equal-area centred on Europe (as in Eurostat's
// ETRS89-LAEA): areas stay comparable, so big countries do not dominate.
const RAD = Math.PI / 180;
const LON0 = 10 * RAD, LAT0 = 52 * RAD;
const LAEA = {
  project([lon, lat]: number[]): number[] {
    const l = lon * RAD - LON0, p = lat * RAD;
    const k = Math.sqrt(2 / (1 + Math.sin(LAT0) * Math.sin(p) + Math.cos(LAT0) * Math.cos(p) * Math.cos(l)));
    const x = k * Math.cos(p) * Math.sin(l);
    const y = k * (Math.cos(LAT0) * Math.sin(p) - Math.sin(LAT0) * Math.cos(p) * Math.cos(l));
    return [x, -y];                                  // screen y grows downwards
  },
  unproject([x, yScreen]: number[]): number[] {
    const y = -yScreen;
    const rho = Math.hypot(x, y);
    if (rho === 0) return [LON0 / RAD, LAT0 / RAD];
    const c = 2 * Math.asin(Math.min(1, rho / 2));
    const lat = Math.asin(Math.cos(c) * Math.sin(LAT0) + (y * Math.sin(c) * Math.cos(LAT0)) / rho);
    const lon = LON0 + Math.atan2(x * Math.sin(c), rho * Math.cos(LAT0) * Math.cos(c) - y * Math.sin(LAT0) * Math.sin(c));
    return [lon / RAD, lat / RAD];
  },
};

const NO_DECAL = { symbol: 'none' };

export interface MapCategory { label: string; color: string; }

/**
 * Choropleth of Europe. Give either `values` (continuous: sequential scale,
 * or diverging around 0 when the range spans zero) or `categories` (country →
 * category label, coloured by `categoryColors`). Countries outside the panel
 * stay neutral grey. Clicking a panel country emits `countryClick`.
 */
@Component({
  selector: 'app-europe-map',
  imports: [NgxEchartsDirective],
  template: `
    <div echarts class="map" [style.height.px]="height()" [options]="options()"
         [loading]="!ready()" (chartClick)="onClick($event)"></div>
    @if (legend().length) {
      <div class="legend" role="list">
        @for (l of legend(); track l.label) {
          <span role="listitem"><i [style.background]="l.color"></i>{{ l.label }}</span>
        }
        <span role="listitem"><i class="none"></i>not in the panel</span>
      </div>
    }
    <p class="src">Boundaries: Natural Earth 1:50m (public domain); equal-area projection.</p>`,
  styles: [`
    .map { width: 100%; }
    .legend { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 12px; color: var(--ink-2); margin-top: 6px; }
    .legend i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -2px; }
    .src { color: var(--muted); font-size: 11px; margin-top: 4px; }
    .legend i.none { border: 1px solid var(--axis);
      background: repeating-linear-gradient(45deg, var(--surface-2) 0 3px, var(--axis) 3px 4px); }
  `],
})
export class EuropeMap {
  private theme = inject(ThemeService);
  protected ready = toSignal(inject(EuropeMapLoader).ready$, { initialValue: false });

  readonly values = input<Record<string, number | null>>();
  readonly categories = input<Record<string, string>>();
  readonly categoryColors = input<Record<string, string>>({});
  /** fixed colour range, e.g. the variable's range over all years */
  readonly range = input<[number, number]>();
  readonly unit = input('');
  readonly selected = input<string[]>([]);
  readonly height = input(440);
  readonly countryClick = output<string>();

  protected legend = computed<MapCategory[]>(() => {
    const cats = this.categories();
    if (!cats) return [];
    const colors = this.categoryColors();
    const order = Object.keys(colors);
    const rank = (l: string) => (order.includes(l) ? order.indexOf(l) : order.length);
    const seen = [...new Set(Object.values(cats))].sort((a, b) => rank(a) - rank(b));
    return seen.map(label => ({ label, color: colors[label] ?? this.theme.tokens().muted }));
  });

  protected onClick(e: { name?: string; data?: unknown }) {
    if (e?.name && e.data) this.countryClick.emit(e.name);
  }

  protected options = computed<EChartsOption>(() => {
    const t = this.theme.tokens();
    if (!this.ready()) return base(t);
    const sel = new Set(this.selected());
    const vals = this.values();
    const cats = this.categories();
    const colors = this.categoryColors();
    const border = (c: string) => (sel.has(c) ? { borderColor: t.ink, borderWidth: 2 } : {});

    const series: Record<string, unknown> = {
      type: 'map', map: 'europe', projection: LAEA, roam: false, selectedMode: false,
      showLegendSymbol: false,
      left: 8, right: 8, top: 8, bottom: vals ? 56 : 8,
      // countries outside the panel: hatched, so "no data" never reads as a value
      itemStyle: { areaColor: t.surface2, borderColor: t.surface, borderWidth: 0.8,
        decal: { symbol: 'rect', symbolSize: 1, dashArrayX: [1, 0], dashArrayY: [1, 4],
          rotation: Math.PI / 4, color: t.axis } },
      emphasis: { label: { show: false }, itemStyle: { areaColor: undefined, borderColor: t.ink, borderWidth: 1.5 } },
      label: { show: false },
    };
    const opts: EChartsOption = {
      ...base(t),
      tooltip: {
        ...(base(t).tooltip as object), trigger: 'item',
        formatter: (p: any) => {
          if (cats) return cats[p.name] ? `<b>${p.name}</b><br>${cats[p.name]}` : `<b>${p.name}</b><br>not in the panel`;
          const v = p.value;
          return typeof v === 'number' && !Number.isNaN(v)
            ? `<b>${p.name}</b><br>${fmt(v, Math.abs(v) >= 100 ? 0 : 2)} ${this.unit()}`
            : `<b>${p.name}</b><br>${vals && p.name in vals ? 'no data this year' : 'not in the panel'}`;
        },
      },
    };

    if (cats) {
      series['data'] = Object.entries(cats).map(([name, label]) => ({
        name, value: 1, itemStyle: { areaColor: colors[label] ?? t.muted, decal: NO_DECAL, ...border(name) },
      }));
    } else if (vals) {
      const finite = Object.values(vals).filter((v): v is number => v !== null && Number.isFinite(v));
      let [lo, hi] = this.range() ?? [Math.min(...finite), Math.max(...finite)];
      const diverging = lo < 0 && hi > 0;
      if (diverging) { const m = Math.max(-lo, hi); lo = -m; hi = m; }
      series['data'] = Object.entries(vals).map(([name, v]) => ({
        name, value: v ?? '-', itemStyle: { decal: v === null ? undefined : NO_DECAL, ...border(name) },
      }));
      opts.visualMap = {
        type: 'continuous', min: lo, max: hi, calculable: false, realtime: false,
        orient: 'horizontal', left: 'center', bottom: 4, itemWidth: 10, itemHeight: 220,
        text: [fmt(hi, Math.abs(hi) >= 100 ? 0 : 1), fmt(lo, Math.abs(lo) >= 100 ? 0 : 1)],
        textStyle: { color: t.ink2, fontSize: 11 },
        inRange: { color: diverging ? t.div : t.seq.slice(1) },
        outOfRange: { color: t.surface2 },
      };
    }
    opts.series = [series as never];
    return opts;
  });
}
