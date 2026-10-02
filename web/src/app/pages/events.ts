import { Component, computed, inject, signal } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import type { EChartsOption } from 'echarts';
import { NgxEchartsDirective } from 'ngx-echarts';
import { ApiService } from '../core/api.service';
import { band, base, categoryAxis, fmt, stars, valueAxis } from '../core/charts';
import { EventStudyRequest, EventStudyResponse } from '../core/models';
import { ThemeService } from '../core/theme.service';

@Component({
  selector: 'app-events',
  imports: [NgxEchartsDirective, FormsModule],
  template: `
  <section class="page">
    <header class="page-head">
      <h1>Staggered event study</h1>
      <p>Callaway–Sant'Anna average treatment effects by years since the event, with a
         bootstrap 95 % band. Pre-event estimates test parallel trends: they should hover
         around zero.</p>
    </header>

    <div class="card controls">
      <label class="field">Outcome
        <select [(ngModel)]="req.outcome">
          @for (v of meta()?.variables ?? []; track v.key) { <option [value]="v.key">{{ v.label }}</option> }
        </select></label>
      <label class="field">Events
        <select [(ngModel)]="req.event_set">
          @for (k of eventSets(); track k) { <option [value]="k">{{ k }}</option> }
          <option value="Custom">Custom…</option>
        </select></label>
      @if (req.event_set === 'Custom') {
        <label class="field">Country:Year, …
          <input type="text" [(ngModel)]="req.custom" placeholder="Poland:2016, Slovakia:2015" size="28"></label>
      }
      <label class="field">Controls
        <select [(ngModel)]="req.control">
          @for (c of meta()?.control_groups ?? []; track c) { <option [value]="c">{{ c }}</option> }
        </select></label>
      <label class="field">Years before / after
        <span><input type="number" min="2" max="6" [(ngModel)]="req.pre">
              <input type="number" min="1" max="8" [(ngModel)]="req.post"></span></label>
      <label class="field">Bootstrap draws
        <select [(ngModel)]="req.n_boot">
          <option [ngValue]="49">49 · quick</option><option [ngValue]="99">99 · default</option>
          <option [ngValue]="199">199 · thesis</option><option [ngValue]="499">499 · slow</option>
        </select></label>
      <label class="toggle"><input type="checkbox" [(ngModel)]="req.detrend"> Remove pre-trends</label>
      <button class="btn primary" type="button" (click)="run()" [disabled]="busy()">
        @if (busy()) { <span class="spinner"></span> Estimating… } @else { Run }
      </button>
    </div>

    @if (error()) { <p class="error card">{{ error() }}</p> }

    <div class="grid-tiles">
      <div class="tile"><div class="label">Average effect after the event</div>
        <div class="value">{{ res() ? f(res()!.overall_post) + '%' : '…' }}</div>
        <div class="hint">{{ res() ? 'SE ' + f(res()!.overall_post_se) + ' · ' + sig() : '' }}</div></div>
      <div class="tile"><div class="label">Pre-trend test (p)</div>
        <div class="value">{{ res() ? f(res()!.pretrend_p, 2) : '…' }}</div>
        <div class="hint">
          @if (res()) {
            <span class="badge" [class.good]="res()!.pretrend_p >= 0.1" [class.bad]="res()!.pretrend_p < 0.1">
              {{ res()!.pretrend_p >= 0.1 ? '✓ parallel trends plausible' : '✕ pre-trends detected' }}</span>
          }</div></div>
      <div class="tile"><div class="label">Treated countries</div>
        <div class="value">{{ res()?.treated?.length ?? '…' }}</div>
        <div class="hint">{{ res()?.controls?.length ?? 0 }} never-treated controls</div></div>
    </div>

    <div class="card">
      <div class="card-head"><h2>Effect on {{ outcomeLabel() }} by years since the event</h2>
        <span class="sub">% difference vs controls</span></div>
      <div echarts class="chart tall" [options]="options()" [loading]="busy()"></div>
      <p class="note">The year before the event (e = −1) is the reference year, so it has no estimate.</p>
      @if (res()) {
        <div class="events">
          <h3>Event dates</h3>
          <div class="chips">
            @for (e of eventList(); track e[0]) { <span class="badge">{{ e[0] }} {{ e[1] }}</span> }
          </div>
        </div>
      }
      <details>
        <summary>Show estimates</summary>
        <div class="table-wrap"><table class="data">
          <thead><tr><th>Years since event</th><th>ATT %</th><th>SE</th><th>95 % band</th><th>Cohorts</th></tr></thead>
          <tbody>
            @for (r of res()?.by_event_time ?? []; track r.e) {
              <tr><td>{{ r.e }}</td><td>{{ f(r.att) }}</td><td>{{ f(r.se) }}</td>
                <td>[{{ f(r.lo) }}, {{ f(r.hi) }}]</td><td>{{ r.n_cohorts }}</td></tr>
            }
          </tbody>
        </table></div>
      </details>
    </div>
  </section>`,
  styles: [`
    .field span { display: flex; gap: 6px; }
    .events { display: grid; gap: 6px; margin-top: 8px; }
  `],
})
export class Events {
  private api = inject(ApiService);
  private theme = inject(ThemeService);
  protected meta = toSignal(this.api.meta());
  protected eventSets = computed(() => Object.keys(this.meta()?.event_sets ?? {}));
  protected req: EventStudyRequest = {
    outcome: 'RD_pct_GDP', event_set: 'R&D tax reforms', custom: '',
    control: 'Not yet treated', detrend: false, pre: 3, post: 5, n_boot: 99,
  };
  protected res = signal<EventStudyResponse | undefined>(undefined);
  protected busy = signal(false);
  protected error = signal('');
  protected f = fmt;

  constructor() { this.run(); }

  protected run() {
    this.busy.set(true);
    this.error.set('');
    this.api.eventStudy({ ...this.req }).subscribe({
      next: r => { this.res.set(r); this.busy.set(false); },
      error: (e: Error) => { this.error.set(e.message); this.busy.set(false); },
    });
  }

  protected outcomeLabel = computed(() => {
    const k = this.res()?.outcome ?? this.req.outcome;
    return this.meta()?.variables.find(v => v.key === k)?.label ?? k;
  });
  protected eventList = computed(() =>
    Object.entries(this.res()?.events ?? {}).sort((a, b) => a[1] - b[1] || a[0].localeCompare(b[0])));
  protected sig = computed(() => {
    const r = this.res();
    if (!r) return '';
    const z = Math.abs(r.overall_post / r.overall_post_se);
    const p = 2 * (1 - normCdf(z));
    return p < 0.1 ? `significant${stars(p)}` : 'not significant';
  });

  protected options = computed<EChartsOption>(() => {
    const t = this.theme.tokens();
    const r = this.res();
    if (!r) return base(t);
    const rows = r.by_event_time;
    const es = rows.map(x => x.e);
    const col = t.series[0];
    const pre = rows.map(x => (x.e < 0 ? x.att : null));
    const post = rows.map(x => (x.e >= 0 ? x.att : null));
    const byE = new Map(rows.map(x => [String(x.e), x]));
    return {
      ...base(t),
      grid: { left: 56, right: 24, top: 60, bottom: 48 },
      legend: { ...(base(t).legend as object), data: ['Before the event', 'After the event'] },
      xAxis: { ...categoryAxis(t, es, 'years since the event (0 = event year)'), boundaryGap: true },
      yAxis: valueAxis(t, '%'),
      tooltip: {
        ...(base(t).tooltip as object), trigger: 'axis',
        axisPointer: { type: 'line', lineStyle: { color: t.axis } },
        formatter: (ps: any) => {
          const x = byE.get(String(ps[0]?.axisValue));
          return x ? `<b>e = ${x.e}</b><br>ATT ${fmt(x.att)} % [${fmt(x.lo)}, ${fmt(x.hi)}]<br>${x.n_cohorts} cohorts` : '';
        },
      },
      series: [
        ...(band(t, 'att', rows.map(x => x.lo), rows.map(x => x.hi), col, 'ci') as any[]),
        { name: 'Before the event', type: 'line', data: pre, symbol: 'circle', symbolSize: 8,
          lineStyle: { width: 2, color: t.muted, type: 'dashed' },
          itemStyle: { color: t.surface, borderColor: t.muted, borderWidth: 2 },
          markLine: { silent: true, symbol: 'none', label: { show: false },
            lineStyle: { color: t.axis, type: 'solid', width: 1 }, data: [{ yAxis: 0 }] } },
        { name: 'After the event', type: 'line', data: post, symbol: 'circle', symbolSize: 8,
          lineStyle: { width: 2, color: col },
          itemStyle: { color: col, borderColor: t.surface, borderWidth: 2 } },
      ],
    } as EChartsOption;
  });
}

/** Standard normal CDF (Abramowitz–Stegun 7.1.26). */
function normCdf(z: number) {
  const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
    * Math.exp(-(z * z) / 2);
  return z >= 0 ? (1 + y) / 2 : (1 - y) / 2;
}
