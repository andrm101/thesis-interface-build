import { Component, computed, inject } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import type { EChartsOption } from 'echarts';
import { NgxEchartsDirective } from 'ngx-echarts';
import { ApiService } from '../core/api.service';
import { base } from '../core/charts';
import { ThemeService } from '../core/theme.service';

@Component({
  selector: 'app-coverage',
  imports: [NgxEchartsDirective],
  template: `
  <section class="page">
    <header class="page-head">
      <h1>Data coverage</h1>
      <p>Share of years {{ cov()?.window?.[0] }}–{{ cov()?.window?.[1] }} with data, by country and
         variable. Gaps limit what each analysis can use; the policy series come from Eurostat,
         the OECD and AMECO (see the data audit).</p>
    </header>
    <div class="card">
      <div echarts class="chart" [style.height.px]="height()" [options]="options()" [loading]="!cov()"></div>
    </div>
  </section>`,
})
export class Coverage {
  private api = inject(ApiService);
  private theme = inject(ThemeService);
  protected cov = toSignal(this.api.coverage());
  protected meta = toSignal(this.api.meta());
  protected height = computed(() => 120 + 24 * (this.cov()?.countries.length ?? 20));

  protected options = computed<EChartsOption>(() => {
    const t = this.theme.tokens();
    const c = this.cov();
    if (!c) return base(t);
    const label = (k: string) => this.meta()?.variables.find(v => v.key === k)?.label ?? k;
    const xs = c.variables.map(label);
    const ys = [...c.countries].reverse();
    const data = c.cells.map(x => [c.variables.indexOf(x.variable), ys.indexOf(x.country),
      Math.round(100 * x.share)]);
    return {
      ...base(t),
      grid: { left: 100, right: 24, top: 150, bottom: 56 },
      xAxis: { type: 'category', data: xs, position: 'top', splitArea: { show: false },
        axisLabel: { color: t.ink2, fontSize: 11, rotate: 40, interval: 0, width: 150, overflow: 'truncate' },
        axisLine: { show: false }, axisTick: { show: false } },
      yAxis: { type: 'category', data: ys, axisLabel: { color: t.ink2, fontSize: 11 },
        axisLine: { show: false }, axisTick: { show: false } },
      visualMap: {
        min: 0, max: 100, calculable: false, orient: 'horizontal', left: 'center', bottom: 8,
        itemWidth: 12, itemHeight: 160, text: ['100 % of years', '0 %'],
        textStyle: { color: t.ink2, fontSize: 11 },
        inRange: { color: t.seq },
      },
      tooltip: { ...(base(t).tooltip as object), trigger: 'item',
        formatter: (p: any) => `<b>${ys[p.value[1]]}</b><br>${xs[p.value[0]]}: ${p.value[2]} % of years` },
      series: [{
        type: 'heatmap', data,
        itemStyle: { borderColor: t.surface, borderWidth: 2, borderRadius: 3 },
        label: { show: false },
        emphasis: { itemStyle: { borderColor: t.ink, borderWidth: 1 } },
      }],
    } as EChartsOption;
  });
}
