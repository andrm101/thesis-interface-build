import { Component, inject } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { ThemeService } from './core/theme.service';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  protected theme = inject(ThemeService);
  protected nav = [
    { path: '/', label: 'Overview', icon: 'M3 12l9-8 9 8M5 10v10h14V10' },
    { path: '/explore', label: 'Explore', icon: 'M3 17l6-6 4 4 8-8M15 7h6v6' },
    { path: '/projections', label: 'Local projections', icon: 'M3 12h4l3-7 4 14 3-7h4' },
    { path: '/events', label: 'Event study', icon: 'M12 3v18M4 15l4-4 4 3 8-8' },
    { path: '/coverage', label: 'Data coverage', icon: 'M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z' },
    { path: '/results', label: 'Results & figures', icon: 'M6 3h9l4 4v14H6zM14 3v5h5M9 13h7M9 17h7' },
  ];
}
