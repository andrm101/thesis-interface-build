import type { EChartsOption } from 'echarts';
import { Group } from './models';
import { Tokens } from './theme.service';

/** Shared chart chrome: recessive grid and axes, text in ink tokens (never
 *  the series colour), one tooltip style. */
export function base(t: Tokens): EChartsOption {
  return {
    backgroundColor: 'transparent',
    textStyle: { fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: t.ink2 },
    animationDuration: 300,
    grid: { left: 56, right: 24, top: 56, bottom: 44, containLabel: false },
    tooltip: {
      backgroundColor: t.surface, borderColor: t.axis, borderWidth: 1,
      textStyle: { color: t.ink, fontSize: 12 },
      extraCssText: 'border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,.12);',
    },
    legend: {
      top: 0, left: 0, icon: 'roundRect', itemWidth: 12, itemHeight: 4,
      textStyle: { color: t.ink2, fontSize: 12 }, inactiveColor: t.axis,
    },
  };
}

export function valueAxis(t: Tokens, name = ''): Record<string, unknown> {
  return {
    type: 'value', name, nameLocation: 'end', nameGap: 12,
    nameTextStyle: { color: t.muted, fontSize: 11, align: 'left' },
    axisLabel: { color: t.muted, fontSize: 11 },
    splitLine: { lineStyle: { color: t.grid, width: 1 } },
    axisLine: { show: false }, axisTick: { show: false },
  };
}

export function categoryAxis(t: Tokens, data: (string | number)[], name = ''): Record<string, unknown> {
  return {
    type: 'category', data, name, nameLocation: 'middle', nameGap: 28,
    nameTextStyle: { color: t.muted, fontSize: 11 },
    axisLabel: { color: t.muted, fontSize: 11 },
    axisLine: { lineStyle: { color: t.axis } }, axisTick: { show: false },
    boundaryGap: false,
  };
}

/** Typology groups keep the same colours everywhere (and in figures.py). */
export function groupColor(t: Tokens, g: Group | string): string {
  return g === 'Innovative' ? t.series[0] : t.series[1];
}

export const fmt = (x: number | null | undefined, d = 2) =>
  x === null || x === undefined || Number.isNaN(x) ? '—' : x.toFixed(d);

export const stars = (p: number) => (p < 0.01 ? '***' : p < 0.05 ? '**' : p < 0.1 ? '*' : '');

/** Confidence band as a stacked area pair (lower edge + band width). */
export function band(t: Tokens, name: string, lo: number[], hi: number[], color: string,
                     stack: string): EChartsOption['series'] {
  return [
    { name: `${name} lo`, type: 'line', data: lo, stack, symbol: 'none',
      stackStrategy: 'all', lineStyle: { opacity: 0 }, silent: true, tooltip: { show: false } },
    { name: `${name} band`, type: 'line', data: hi.map((h, i) => h - lo[i]), stack,
      stackStrategy: 'all', symbol: 'none', lineStyle: { opacity: 0 }, silent: true,
      tooltip: { show: false },
      areaStyle: { color, opacity: 0.16 } },
  ] as EChartsOption['series'];
}
