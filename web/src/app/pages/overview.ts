import { Component, computed, inject } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { RouterLink } from '@angular/router';
import type { EChartsOption } from 'echarts';
import { NgxEchartsDirective } from 'ngx-echarts';
import { ApiService } from '../core/api.service';
import { base, categoryAxis, groupColor, valueAxis } from '../core/charts';
import { ThemeService } from '../core/theme.service';

const FINDINGS = [
  { id: 'B2', text: 'Two R&D regimes are real: K-Means on R&D levels reproduces the thesis typology (Innovative vs Emerging).' },
  { id: 'C1', text: 'Catch-up happens from below: β-convergence holds in the Emerging group only.' },
  { id: 'B6', text: 'Productivity drives R&D more than the reverse: Granger causality runs from income to R&D.' },
  { id: 'B13', text: 'With the feedback removed (system GMM), R&D pays off mainly in the leaders, but the result is fragile.' },
  { id: 'B9', text: 'EU accession cannot be separated from catch-up already under way (pre-trends fail).' },
  { id: 'D2', text: 'R&D tax reforms raise R&D intensity by about 3–5 % (not significant); productivity does not move.' },
  { id: 'D5', text: 'Public R&D budgets partly crowd out private R&D: multiplier 0.4–0.6, crowding-out persists in the leaders.' },
  { id: 'D4', text: 'Fiscal tightening costs output for about two years, with no lasting damage to productivity or R&D.' },
];

@Component({
  selector: 'app-overview',
  imports: [NgxEchartsDirective, RouterLink],
  template: `
  <section class="page">
    <header class="page-head">
      <h1>R&amp;D, productivity and policy in the EU</h1>
      <p>Twenty-four EU economies, 2000–2023. Every chart is computed live by the same Python
         functions that produce <code>results/RESULTS.md</code> and the thesis figures.</p>
    </header>

    <div class="grid-tiles">
      <div class="tile"><div class="label">Countries</div>
        <div class="value">{{ meta()?.countries?.length ?? '…' }}</div>
        <div class="hint">{{ meta()?.screen }}</div></div>
      <div class="tile"><div class="label">Years</div>
        <div class="value">{{ meta() ? meta()!.years[0] + '–' + meta()!.years[1] : '…' }}</div>
        <div class="hint">annual panel</div></div>
      <div class="tile"><div class="label">Variables</div>
        <div class="value">{{ meta()?.variables?.length ?? '…' }}</div>
        <div class="hint">levels, R&amp;D inputs, policy</div></div>
      <div class="tile"><div class="label">R&amp;D tax reforms</div>
        <div class="value">{{ reforms() ?? '…' }}</div>
        <div class="hint">dated from OECD subsidy jumps</div></div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-head"><h2>R&amp;D typology</h2>
          <span class="sub">{{ typSub() }}</span></div>
        <div echarts class="chart" [options]="typOptions()" [loading]="!typ()"></div>
        <p class="note">Principal components of country-mean R&amp;D intensity, researchers,
          patents and tertiary share; colour = K-Means cluster.</p>
      </div>
      <div class="card">
        <div class="card-head"><h2>Convergence clubs</h2>
          <span class="sub">{{ clubSub() }}</span></div>
        <div echarts class="chart" [options]="clubOptions()" [loading]="!clubs()"></div>
        <p class="note">Relative transition paths of log output per worker (Phillips–Sul);
          1 = the cross-country average.</p>
      </div>
    </div>

    <div class="card">
      <div class="card-head"><h2>Key findings</h2>
        <a routerLink="/results" class="sub">All results →</a></div>
      <ol class="findings">
        @for (f of findings; track f.id) {
          <li><span>{{ f.text }}</span>
            <a [routerLink]="['/results']" [queryParams]="{ id: f.id }" class="badge">{{ f.id }}</a></li>
        }
      </ol>
    </div>
  </section>`,
  styles: [`
    .findings { margin: 0; padding-left: 20px; display: grid; gap: 8px; }
    .findings li { color: var(--ink); }
    .findings li span { margin-right: 8px; }
    code { font-size: 12.5px; background: var(--surface-2); padding: 1px 5px; border-radius: 4px; }
  `],
})
export class Overview {
  private api = inject(ApiService);
  private theme = inject(ThemeService);
  protected findings = FINDINGS;
  protected meta = toSignal(this.api.meta());
  protected typ = toSignal(this.api.typology());
  protected clubs = toSignal(this.api.clubs());

  protected reforms = computed(() => {
    const ev = this.meta()?.event_sets['R&D tax reforms'];
    return ev ? Object.keys(ev).length : undefined;
  });
  protected typSub = computed(() => {
    const t = this.typ();
    return t ? `silhouette ${t.silhouette.toFixed(2)} · agreement with thesis ${(t.agreement * 100).toFixed(0)} %` : '';
  });
  protected clubSub = computed(() => {
    const c = this.clubs();
    return c ? `${c.clubs.length} clubs · full-sample log-t ${c.full_t.toFixed(1)}` : '';
  });

  protected typOptions = computed<EChartsOption>(() => {
    const t = this.theme.tokens();
    const d = this.typ();
    if (!d) return base(t);
    const ex = d.explained.map(e => Math.round(e * 100));
    const series = (['Innovative', 'Emerging'] as const).map(g => ({
      name: g, type: 'scatter' as const, symbolSize: 11,
      itemStyle: { color: groupColor(t, g), borderColor: t.surface, borderWidth: 2 },
      data: d.points.filter(p => p.cluster === g).map(p => ({ value: [p.x, p.y], name: p.country, group: p.group })),
      label: { show: true, formatter: '{b}', position: 'right' as const, fontSize: 11, color: t.ink2 },
      labelLayout: { hideOverlap: true },
      emphasis: { focus: 'self' as const, label: { color: t.ink, fontWeight: 'bold' as const } },
    }));
    return {
      ...base(t),
      grid: { left: 48, right: 84, top: 56, bottom: 44 },
      xAxis: { ...valueAxis(t), name: `PC1 (${ex[0]} %)`, nameLocation: 'middle', nameGap: 28, splitLine: { show: false } },
      yAxis: { ...valueAxis(t, `PC2 (${ex[1]} %)`) },
      tooltip: { ...(base(t).tooltip as object), trigger: 'item',
        formatter: (p: any) => `<b>${p.name}</b><br>cluster: ${p.seriesName}<br>thesis group: ${p.data.group}` },
      series,
    } as EChartsOption;
  });

  protected clubOptions = computed<EChartsOption>(() => {
    const t = this.theme.tokens();
    const d = this.clubs();
    if (!d) return base(t);
    const name = (k: number) => (k === 0 ? 'Non-convergent' : `Club ${k}`);
    const color = (k: number) => (k === 0 ? t.muted : t.series[(k - 1) % 8]);
    const legend = [...d.clubs.map(c => name(c.club)), ...(d.divergent.length ? [name(0)] : [])];
    return {
      ...base(t),
      legend: { ...(base(t).legend as object), data: legend },
      xAxis: categoryAxis(t, d.years),
      yAxis: { ...valueAxis(t), scale: true },
      tooltip: { ...(base(t).tooltip as object), trigger: 'item',
        formatter: (p: any) => `<b>${p.data.country}</b> · ${p.seriesName}<br>${p.name}: ${(+p.value).toFixed(3)}` },
      series: d.paths.map(p => ({
        name: name(p.club), type: 'line' as const, showSymbol: false, symbolSize: 6,
        lineStyle: { width: 1.5, color: color(p.club) }, itemStyle: { color: color(p.club) },
        emphasis: { focus: 'series' as const, lineStyle: { width: 2.5 } },
        data: p.values.map(v => ({ value: v, country: p.country })),
      })),
    } as EChartsOption;
  });
}
