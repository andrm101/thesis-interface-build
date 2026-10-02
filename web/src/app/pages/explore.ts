import { Component, DestroyRef, computed, effect, inject, signal } from '@angular/core';
import { toObservable, toSignal } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import type { EChartsOption } from 'echarts';
import { NgxEchartsDirective } from 'ngx-echarts';
import { combineLatest, debounceTime, switchMap } from 'rxjs';
import { ApiService } from '../core/api.service';
import { base, categoryAxis, fmt, groupColor, valueAxis } from '../core/charts';
import { EuropeMap } from '../core/europe-map';
import { ThemeService } from '../core/theme.service';

const MAX_HIGHLIGHT = 6;

@Component({
  selector: 'app-explore',
  imports: [NgxEchartsDirective, FormsModule, EuropeMap],
  template: `
  <section class="page">
    <header class="page-head">
      <h1>Explore the panel</h1>
      <p>Pick a variable, then highlight up to {{ max }} countries. Every other country stays
         in the background for context.</p>
    </header>

    <div class="card controls">
      <label class="field">Variable
        <select [ngModel]="variable()" (ngModelChange)="variable.set($event)">
          @for (v of meta()?.variables ?? []; track v.key) {
            <option [value]="v.key">{{ v.label }} ({{ v.unit }})</option>
          }
        </select>
      </label>
      <label class="field">Year (map and ranking): <b>{{ year() }}</b>
        <input type="range" [min]="years()[0]" [max]="years()[1]" [ngModel]="year()"
               (ngModelChange)="stop(); year.set(+$event)">
      </label>
      <button class="btn" type="button" (click)="playing() ? stop() : play()"
              [attr.aria-label]="playing() ? 'Pause the animation' : 'Animate the years'">
        {{ playing() ? '❚❚ Pause' : '▶ Play years' }}</button>
      <button class="btn" type="button" (click)="clear()">Clear highlights</button>
    </div>

    <div class="card">
      <div class="card-head"><h2>Highlighted countries</h2>
        <span class="sub">{{ selected().length }} / {{ max }} selected</span></div>
      @for (g of ['Innovative', 'Emerging']; track g) {
        <div class="group-row">
          <h3>{{ g }}</h3>
          <div class="chips">
            @for (c of byGroup()[g] ?? []; track c) {
              <button type="button" class="chip" [class.on]="slotOf(c) !== undefined"
                      [style.color]="slotOf(c) !== undefined ? colorOf(c) : null"
                      [attr.aria-pressed]="slotOf(c) !== undefined" (click)="toggle(c)">
                @if (slotOf(c) !== undefined) { <span class="dot" [style.background]="colorOf(c)"></span> }
                {{ c }}
              </button>
            }
          </div>
        </div>
      }
    </div>

    <div class="card">
      <div class="card-head"><h2>{{ label() }} over time</h2><span class="sub">{{ unit() }}</span></div>
      @if (error()) { <p class="error">{{ error() }}</p> }
      <div echarts class="chart tall" [options]="lineOptions()" [loading]="!series()"></div>
      <details>
        <summary>Show data table</summary>
        <div class="table-wrap"><table class="data">
          <thead><tr><th>Year</th>@for (c of selected(); track c) { <th>{{ c }}</th> }</tr></thead>
          <tbody>
            @for (row of tableRows(); track row.year) {
              <tr><td>{{ row.year }}</td>@for (v of row.values; track $index) { <td>{{ fmtv(v) }}</td> }</tr>
            }
          </tbody>
        </table></div>
      </details>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-head"><h2>{{ label() }}, {{ year() }}</h2><span class="sub">{{ unit() }}</span></div>
        <app-europe-map [values]="mapValues()" [range]="mapRange()" [unit]="unit()"
                        [selected]="selected()" [height]="460" (countryClick)="toggle($event)" />
        <p class="note">Colour scale fixed over 2000–2023, so years are comparable. Click a
          country to highlight it.</p>
      </div>
      <div class="card">
        <div class="card-head"><h2>Ranking in {{ year() }}</h2><span class="sub">{{ unit() }}</span></div>
        <div echarts class="chart tall" [options]="barOptions()" [loading]="!snapshot()"></div>
      </div>
    </div>
  </section>`,
  styles: [`
    .group-row { display: grid; grid-template-columns: 90px 1fr; gap: 12px; align-items: start; padding: 6px 0; }
    .group-row h3 { padding-top: 4px; }
    @media (max-width: 700px) { .group-row { grid-template-columns: 1fr; gap: 4px; } }
  `],
})
export class Explore {
  private api = inject(ApiService);
  private theme = inject(ThemeService);
  protected max = MAX_HIGHLIGHT;

  protected meta = toSignal(this.api.meta());
  protected variable = signal('RD_pct_GDP');
  protected year = signal(2023);
  /** country → palette slot; a colour stays with its country while selected */
  private slots = signal(new Map<string, number>([['Poland', 0], ['Germany', 1], ['Romania', 2]]));
  protected selected = computed(() => [...this.slots().keys()]);
  protected error = signal('');

  protected years = computed<[number, number]>(() => this.meta()?.years ?? [2000, 2023]);
  protected varInfo = computed(() => this.meta()?.variables.find(v => v.key === this.variable()));
  protected label = computed(() => this.varInfo()?.label ?? this.variable());
  protected unit = computed(() => this.varInfo()?.unit ?? '');
  protected byGroup = computed(() => {
    const out: Record<string, string[]> = {};
    for (const c of this.meta()?.countries ?? []) (out[c.group] ??= []).push(c.name);
    return out;
  });

  protected series = toSignal(toObservable(this.variable).pipe(
    switchMap(v => this.api.series(v))));
  protected snapshot = toSignal(combineLatest([toObservable(this.variable), toObservable(this.year)]).pipe(
    debounceTime(150), switchMap(([v, y]) => this.api.snapshot(v, y))));

  protected playing = signal(false);
  private timer: ReturnType<typeof setInterval> | undefined;

  /** values for the map: every panel country, in the selected year */
  protected mapValues = computed(() => {
    const d = this.series();
    if (!d) return undefined;
    const i = d.years.indexOf(this.year());
    return Object.fromEntries(d.series.map(s => [s.country, i >= 0 ? s.values[i] : null]));
  });
  protected mapRange = computed<[number, number] | undefined>(() => {
    const all = (this.series()?.series ?? []).flatMap(s => s.values)
      .filter((v): v is number => v !== null && Number.isFinite(v));
    return all.length ? [Math.min(...all), Math.max(...all)] : undefined;
  });

  protected play() {
    const [first, last] = this.years();
    if (this.year() >= last) this.year.set(first);
    this.playing.set(true);
    this.timer = setInterval(() => {
      if (this.year() >= this.years()[1]) { this.stop(); return; }
      this.year.update(y => y + 1);
    }, 700);
  }
  protected stop() {
    clearInterval(this.timer);
    this.playing.set(false);
  }

  constructor() {
    inject(DestroyRef).onDestroy(() => this.stop());
    effect(() => { const y = this.years(); if (this.year() > y[1]) this.year.set(y[1]); });
  }

  protected slotOf(c: string) { return this.slots().get(c); }
  protected colorOf(c: string) { return this.theme.tokens().series[this.slots().get(c) ?? 0]; }
  protected fmtv(v: number | null) { return fmt(v, Math.abs(v ?? 0) >= 100 ? 0 : 2); }

  protected toggle(c: string) {
    const m = new Map(this.slots());
    if (m.has(c)) m.delete(c);
    else {
      if (m.size >= MAX_HIGHLIGHT) return;
      const used = new Set(m.values());
      let s = 0;
      while (used.has(s)) s++;
      m.set(c, s);
    }
    this.slots.set(m);
  }
  protected clear() { this.slots.set(new Map()); }

  protected tableRows = computed(() => {
    const d = this.series();
    if (!d) return [];
    const by = new Map(d.series.map(s => [s.country, s.values]));
    return d.years.map((y, i) => ({ year: y, values: this.selected().map(c => by.get(c)?.[i] ?? null) }));
  });

  protected lineOptions = computed<EChartsOption>(() => {
    const t = this.theme.tokens();
    const d = this.series();
    if (!d) return base(t);
    const slots = this.slots();
    const bg = d.series.filter(s => !slots.has(s.country)).map(s => ({
      name: s.country, type: 'line' as const, data: s.values, showSymbol: false, silent: true,
      lineStyle: { width: 1, color: t.axis }, z: 1,
    }));
    const fg = d.series.filter(s => slots.has(s.country)).map(s => {
      const col = t.series[slots.get(s.country)!];
      return {
        name: s.country, type: 'line' as const, data: s.values, showSymbol: false, symbolSize: 8,
        lineStyle: { width: 2, color: col }, itemStyle: { color: col, borderColor: t.surface, borderWidth: 2 },
        endLabel: { show: true, formatter: '{a}', color: t.ink2, fontSize: 11, distance: 6 },
        emphasis: { focus: 'series' as const }, z: 3,
      };
    });
    const unit = this.unit();
    return {
      ...base(t),
      grid: { left: 64, right: 96, top: 32, bottom: 40 },
      legend: { show: false },
      xAxis: categoryAxis(t, d.years),
      yAxis: { ...valueAxis(t, unit), scale: true },
      tooltip: {
        ...(base(t).tooltip as object), trigger: 'axis',
        axisPointer: { type: 'line', lineStyle: { color: t.axis } },
        formatter: (ps: any) => {
          const rows = (ps as any[]).filter(p => slots.has(p.seriesName) && p.value != null)
            .sort((a, b) => b.value - a.value)
            .map(p => `${p.marker} ${p.seriesName}: <b>${fmt(p.value, Math.abs(p.value) >= 100 ? 0 : 2)}</b>`);
          return `<b>${ps[0]?.axisValue}</b><br>` + (rows.join('<br>') || '<span style="opacity:.7">no country highlighted</span>');
        },
      },
      series: [...bg, ...fg],
    } as EChartsOption;
  });

  protected barOptions = computed<EChartsOption>(() => {
    const t = this.theme.tokens();
    const d = this.snapshot();
    if (!d) return base(t);
    const rows = [...d.rows].reverse();            // largest at the top
    const names = rows.map(r => r.country);
    const mk = (g: string) => ({
      name: g, type: 'bar' as const, barWidth: '62%', barGap: '-100%',
      itemStyle: { color: groupColor(t, g), borderRadius: [0, 4, 4, 0] },
      data: rows.map(r => (r.group === g ? r.value : null)),
    });
    return {
      ...base(t),
      grid: { left: 96, right: 56, top: 64, bottom: 24 },
      legend: { ...(base(t).legend as object), data: ['Innovative', 'Emerging'] },
      xAxis: { ...valueAxis(t), position: 'top' },
      yAxis: { type: 'category', data: names, axisTick: { show: false },
        axisLine: { lineStyle: { color: t.axis } },
        axisLabel: { color: t.ink2, fontSize: 11 } },
      tooltip: { ...(base(t).tooltip as object), trigger: 'item',
        formatter: (p: any) => `<b>${p.name}</b> (${p.seriesName})<br>${fmt(p.value, Math.abs(p.value) >= 100 ? 0 : 2)} ${this.unit()}` },
      series: [mk('Innovative'), mk('Emerging')],
    } as EChartsOption;
  });
}
