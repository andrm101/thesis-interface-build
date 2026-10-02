import { HttpClient, HttpErrorResponse, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, catchError, shareReplay, throwError } from 'rxjs';
import {
  ClubsResponse, CoverageResponse, EventStudyRequest, EventStudyResponse, Figure,
  LPRequest, LPResponse, Meta, ResultDetail, ResultSection, SeriesResponse,
  SnapshotResponse, TypologyResponse,
} from './models';

/** Typed client for api/main.py. Same-origin in production; the dev server
 *  proxies /api and /figures to :8000 (proxy.conf.json). */
@Injectable({ providedIn: 'root' })
export class ApiService {
  private http = inject(HttpClient);
  private cache = new Map<string, Observable<unknown>>();

  private get<T>(url: string, params?: Record<string, string | number>): Observable<T> {
    const key = url + JSON.stringify(params ?? {});
    if (!this.cache.has(key)) {
      const p = new HttpParams({ fromObject: params ?? {} });
      this.cache.set(key, this.http.get<T>(url, { params: p }).pipe(
        catchError(toMessage), shareReplay(1)));
    }
    return this.cache.get(key) as Observable<T>;
  }

  meta() { return this.get<Meta>('/api/meta'); }
  series(variable: string) { return this.get<SeriesResponse>('/api/series', { variable }); }
  snapshot(variable: string, year: number) {
    return this.get<SnapshotResponse>('/api/snapshot', { variable, year });
  }
  typology() { return this.get<TypologyResponse>('/api/typology'); }
  clubs(variable = 'Y_per_worker') { return this.get<ClubsResponse>('/api/clubs', { variable }); }
  coverage() { return this.get<CoverageResponse>('/api/coverage'); }
  results() { return this.get<ResultSection[]>('/api/results'); }
  result(id: string) { return this.get<ResultDetail>(`/api/results/${id}`); }
  figures() { return this.get<Figure[]>('/api/figures'); }

  localProjections(req: LPRequest) {
    return this.http.post<LPResponse>('/api/local-projections', req).pipe(catchError(toMessage));
  }
  eventStudy(req: EventStudyRequest) {
    return this.http.post<EventStudyResponse>('/api/event-study', req).pipe(catchError(toMessage));
  }
}

function toMessage(err: HttpErrorResponse) {
  const detail = err.error?.detail;
  const msg = typeof detail === 'string' ? detail
    : err.status === 0 ? 'The API is not reachable — is `uvicorn api.main:app` running?'
    : `Request failed (${err.status})`;
  return throwError(() => new Error(msg));
}
