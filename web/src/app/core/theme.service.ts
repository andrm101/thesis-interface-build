import { Injectable, computed, effect, signal } from '@angular/core';

export type Mode = 'light' | 'dark';

/** Chart colours read back from the CSS tokens in styles.css, so charts and
 *  page chrome always use the same light/dark steps. */
export interface Tokens {
  surface: string; ink: string; ink2: string; muted: string; grid: string; axis: string;
  surface2: string; series: string[]; seq: string[]; div: string[];
}

const KEY = 'eu-panel-theme';

@Injectable({ providedIn: 'root' })
export class ThemeService {
  readonly mode = signal<Mode>(initialMode());
  readonly tokens = computed<Tokens>(() => {
    this.mode();                       // re-read the CSS variables on every switch
    const cs = getComputedStyle(document.documentElement);
    const v = (n: string) => cs.getPropertyValue(n).trim();
    return {
      surface: v('--surface-1'), ink: v('--ink'), ink2: v('--ink-2'), muted: v('--muted'),
      grid: v('--grid'), axis: v('--axis'),
      series: [1, 2, 3, 4, 5, 6, 7, 8].map(i => v(`--s${i}`)),
      seq: [0, 1, 2, 3, 4, 5].map(i => v(`--seq-${i}`)),
      surface2: v('--surface-2'),
      div: ['--div-neg', '--div-neg-2', '--div-mid', '--div-pos-2', '--div-pos'].map(v),
    };
  });

  constructor() {
    apply(this.mode());
    effect(() => apply(this.mode()));
  }

  toggle() {
    const next: Mode = this.mode() === 'dark' ? 'light' : 'dark';
    apply(next);                       // before the signal, so tokens read the new values
    this.mode.set(next);
    try { localStorage.setItem(KEY, next); } catch { /* storage unavailable */ }
  }
}

function apply(mode: Mode) {
  document.documentElement.dataset['theme'] = mode;
}

function initialMode(): Mode {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === 'light' || saved === 'dark') return saved;
  } catch { /* storage unavailable */ }
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}
