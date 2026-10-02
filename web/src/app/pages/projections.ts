import { Component, computed, inject, signal } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import type { EChartsOption } from 'echarts';
import { NgxEchartsDirective } from 'ngx-echarts';
import { ApiService } from '../core/api.service';
import { band, base, categoryAxis, fmt, groupColor, stars, valueAxis } from '../core/charts';
import { LPRequest, LPResponse } from '../core/models';
import { ThemeService } from '../core/theme.service';

const PRESETS: { id: string; label: string; req: LPRequest }[] = [
  { id: 'D5', label: 'Public R&D budget → R&D intensity (D5)',
    req: { outcome: 'RD_pct_GDP', shock: 'GBARD_pct_GDP', horizons: 5, lags: 1, levels: true, split: true } },
  { id: 'D5b', label: 'Public R&D budget → privately financed R&D (D5)',
    req: { outcome: 'NonGBARD_RD_pct_GDP', shock: 'GBARD_pct_GDP', horizons: 5, lags: 1, levels: true, split: true } },
  { id: 'D4', label: 'Cyclically adjusted tightening → output per worker (D4)',
    req: { outcome: 'Y_per_worker', shock: 'CAB_pct_potGDP', horizons: 5, lags: 1, levels: false, split: false } },
  { id: 'D4h', label: 'Headline balance → output per worker (D4)',
    req: { outcome: 'Y_per_worker', shock: 'Gov_balance_pct_GDP', horizons: 5, lags: 1, levels: false, split: false } },
  { id: 'B8', label: 'R&D intensity → output per worker, by group (B8)',
    req: { outcome: 'Y_per_worker', shock: 'RD_pct_GDP', horizons: 5, lags: 2, levels: false, split: true } },
];

@Component({
  selector: 'app-projections',
  imports: [NgxEchartsDirective, FormsModule],
  template: `
  <section class="page">
    <header class="page-head">
      <h1>Local projections</h1>
      <p>Response of an outcome <i>h</i> years after a one-unit change in a shock variable,
         with country and year fixed effects and country-clustered errors (Jordà, 2005).
         Shaded bands are 95 % confidence intervals.</p>
    </header>

    <div class="card">
      <div class="chips presets">
        @for (p of presets; track p.id) {
          <button type="button" class="chip" [class.on]="preset() === p.id" (click)="apply(p.id)">{{ p.label }}</button>
        }
      </div>
      <div class="controls">
        <label class="field">Outcome
          <select [(ngModel)]="req.outcome" (ngModelChange)="preset.set('')">
            @for (v of meta()?.variables ?? []; track v.key) { <option [value]="v.key">{{ v.label }}</option> }
          </select></label>
        <label class="field">Shock (one unit)
          <select [(ngModel)]="req.shock" (ngModelChange)="preset.set('')">
            @for (v of shocks(); track v.key) { <option [value]="v.key">{{ v.label }} ({{ v.unit }})</option> }
          </select></label>
        <label class="field">Horizons
          <input type="number" min="1" max="10" [(ngModel)]="req.horizons"></label>
        <label class="field">Lags
          <input type="number" min="0" max="3" [(ngModel)]="req.lags"></label>
        <label class="toggle"><input type="checkbox" [(ngModel)]="req.levels"> Response in levels</label>
        <label class="toggle"><input type="checkbox" [(ngModel)]="req.split"> Split by group</label>
        <button class="btn primary" type="button" (click)="run()" [disabled]="busy()">
          @if (busy()) { <span class="spinner"></span> } Run
        </button>
      </div>
    </div>

    <div class="card">
      <div class="card-head"><h2>{{ title() }}</h2><span class="sub">{{ subtitle() }}</span></div>
      @if (error()) { <p class="error">{{ error() }}</p> }
      <div echarts class="chart tall" [options]="options()" [loading]="busy()"></div>
      <p class="note">{{ reading() }}</p>
      <details>
        <summary>Show estimates</summary>
        <div class="table-wrap"><table class="data">
          <thead><tr><th>Sample</th><th>h</th><th>β</th><th>SE</th><th>95 % CI</th><th>p</th><th>n</th></tr></thead>
          <tbody>
            @for (r of res()?.rows ?? []; track $index) {
              <tr><td>{{ r.state }}</td><td>{{ r.h }}</td><td>{{ f(r.beta) }}{{ s(r.p) }}</td><td>{{ f(r.se) }}</td>
                <td>[{{ f(r.lo) }}, {{ f(r.hi) }}]</td><td>{{ f(r.p, 3) }}</td><td>{{ r.n }}</td></tr>
            }
          </tbody>
        </table></div>
      </details>
    </div>
  </section>`,
  styles: [`.presets { margin-bottom: 14px; }`],
})
export class Projections {
  private api = inject(ApiService);
  private theme = inject(ThemeService);
  protected presets = PRESETS;
  protected meta = toSignal(this.api.meta());
  protected shocks = computed(() => (this.meta()?.variables ?? []).filter(v => v.shock));
  protected req: LPRequest = { ...PRESETS[0].req };
  protected preset = signal(PRESETS[0].id);
  protected res = signal<LPResponse | undefined>(undefined);
  protected busy = signal(false);
  protected error = signal('');
  protected f = fmt;
  protected s = stars;

  constructor() { this.run(); }

  protected apply(id: string) {
    const p = PRESETS.find(x => x.id === id)!;
    this.req = { ...p.req };
    this.preset.set(id);
    this.run();
  }

  protected run() {
    this.busy.set(true);
    this.error.set('');
    this.api.localProjections({ ...this.req }).subscribe({
      next: r => { this.res.set(r); this.busy.set(false); },
      error: (e: Error) => { this.error.set(e.message); this.busy.set(false); },
    });
  }

  private label(key?: string) {
    return this.meta()?.variables.find(v => v.key === key)?.label ?? key ?? '';
  }
  protected title = computed(() => {
    const r = this.res();
    return r ? `Response of ${this.label(r.outcome)} to ${this.label(r.shock)}` : 'Impulse response';
  });
  protected subtitle = computed(() => {
    const r = this.res();
    return r ? `response: ${r.response_unit} · shock: +1 ${r.shock_unit}` : '';
  });
  protected reading = computed(() => {
    const r = this.res();
    if (!r) return '';
    return r.levels
      ? `β_h is the change in the outcome, in ${r.response_unit}, h years after a one-unit rise in the shock.`
      : `β_h is the % change in the outcome h years after a one-unit rise in the shock.`;
  });

  protected options = computed<EChartsOption>(() => {
    const t = this.theme.tokens();
    const r = this.res();
    if (!r) return base(t);
    const states = [...new Set(r.rows.map(x => x.state))];
    const hs = [...new Set(r.rows.map(x => x.h))];
    const col = (st: string) => (st === 'All' ? t.series[0] : groupColor(t, st));
    const series: any[] = [];
    states.forEach((st, i) => {
      const rows = r.rows.filter(x => x.state === st);
      series.push(...(band(t, st, rows.map(x => x.lo), rows.map(x => x.hi), col(st), `ci${i}`) as any[]));
      series.push({
        name: st, type: 'line', data: rows.map(x => x.beta), symbol: 'circle', symbolSize: 8,
        lineStyle: { width: 2, color: col(st) },
        itemStyle: { color: col(st), borderColor: t.surface, borderWidth: 2 },
        markLine: i ? undefined : { silent: true, symbol: 'none', label: { show: false },
          lineStyle: { color: t.axis, type: 'solid', width: 1 }, data: [{ yAxis: 0 }] },
      });
    });
    const byKey = new Map(r.rows.map(x => [`${x.state}|${x.h}`, x]));
    return {
      ...base(t),
      grid: { left: 64, right: 24, top: 60, bottom: 48 },
      legend: { ...(base(t).legend as object), show: states.length > 1, data: states },
      xAxis: { ...categoryAxis(t, hs, 'years after the shock (h)'), boundaryGap: true },
      yAxis: valueAxis(t, r.levels ? r.response_unit : '% change'),
      tooltip: {
        ...(base(t).tooltip as object), trigger: 'axis',
        axisPointer: { type: 'line', lineStyle: { color: t.axis } },
        formatter: (ps: any) => {
          const h = ps[0]?.axisValue;
          const lines = states.map(st => {
            const x = byKey.get(`${st}|${h}`);
            return x ? `<span style="color:${col(st)}">●</span> ${st}: <b>${fmt(x.beta)}${stars(x.p)}</b> [${fmt(x.lo)}, ${fmt(x.hi)}]` : '';
          });
          return `<b>h = ${h}</b><br>${lines.join('<br>')}`;
        },
      },
      series,
    } as EChartsOption;
  });
}
