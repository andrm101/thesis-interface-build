// Response shapes of the FastAPI back end (api/main.py).

export type Group = 'Innovative' | 'Emerging';

export interface Variable {
  key: string;
  label: string;
  unit: string;
  coverage: number;
  shock: boolean;
}

export interface Meta {
  countries: { name: string; group: Group }[];
  years: [number, number];
  variables: Variable[];
  event_sets: Record<string, Record<string, number>>;
  control_groups: string[];
  screen: string;
}

export interface SeriesResponse {
  variable: string;
  years: number[];
  series: { country: string; group: Group; values: (number | null)[] }[];
}

export interface SnapshotResponse {
  variable: string;
  year: number;
  rows: { country: string; group: Group; value: number }[];
}

export interface TypologyResponse {
  points: { country: string; x: number; y: number; cluster: Group; group: Group }[];
  loadings: { variable: string; pc1: number; pc2: number }[];
  explained: [number, number];
  silhouette: number;
  agreement: number;
}

export interface ClubsResponse {
  variable: string;
  years: number[];
  full_t: number;
  converges: boolean;
  clubs: { club: number; t: number; members: string[] }[];
  divergent: string[];
  paths: { country: string; club: number; values: (number | null)[] }[];
}

export interface LPRequest {
  outcome: string;
  shock: string;
  horizons: number;
  lags: number;
  levels: boolean;
  split: boolean;
}

export interface LPRow {
  h: number; state: string; beta: number; se: number;
  lo: number; hi: number; p: number; n: number;
}

export interface LPResponse {
  outcome: string;
  shock: string;
  levels: boolean;
  response_unit: string;
  shock_unit: string;
  rows: LPRow[];
}

export interface EventStudyRequest {
  outcome: string;
  event_set: string;
  custom: string;
  control: string;
  detrend: boolean;
  pre: number;
  post: number;
  n_boot: number;
}

export interface EventStudyResponse {
  outcome: string;
  events: Record<string, number>;
  by_event_time: { e: number; att: number; se: number; lo: number; hi: number; n_cohorts: number }[];
  overall_post: number;
  overall_post_se: number;
  pretrend_p: number;
  treated: string[];
  controls: string[];
}

export interface CoverageResponse {
  window: [number, number];
  variables: string[];
  countries: string[];
  cells: { country: string; variable: string; share: number }[];
}

export interface ResultSection { id: string; title: string; track: string; }
export interface ResultDetail extends ResultSection { markdown: string; }
export interface Figure { name: string; title: string; png: string; pdf: string; }
