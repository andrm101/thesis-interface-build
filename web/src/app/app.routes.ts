import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', loadComponent: () => import('./pages/overview').then(m => m.Overview),
    title: 'Overview · EU Innovation Panel' },
  { path: 'explore', loadComponent: () => import('./pages/explore').then(m => m.Explore),
    title: 'Explore · EU Innovation Panel' },
  { path: 'projections', loadComponent: () => import('./pages/projections').then(m => m.Projections),
    title: 'Local projections · EU Innovation Panel' },
  { path: 'events', loadComponent: () => import('./pages/events').then(m => m.Events),
    title: 'Event study · EU Innovation Panel' },
  { path: 'coverage', loadComponent: () => import('./pages/coverage').then(m => m.Coverage),
    title: 'Data coverage · EU Innovation Panel' },
  { path: 'results', loadComponent: () => import('./pages/results').then(m => m.Results),
    title: 'Results · EU Innovation Panel' },
  { path: 'assistant', loadComponent: () => import('./pages/assistant').then(m => m.AssistantPage),
    title: 'Assistant · EU Innovation Panel' },
  { path: '**', redirectTo: '' },
];
