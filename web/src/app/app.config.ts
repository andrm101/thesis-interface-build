import { ApplicationConfig, provideBrowserGlobalErrorListeners } from '@angular/core';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { provideRouter, withHashLocation } from '@angular/router';
import { provideEchartsCore } from 'ngx-echarts';
import { routes } from './app.routes';

export const appConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    // hash URLs (#/explore) so any static server — FastAPI included — works
    // without rewrite rules
    provideRouter(routes, withHashLocation()),
    provideHttpClient(withFetch()),
    provideEchartsCore({ echarts: () => import('echarts') }),
  ],
};
