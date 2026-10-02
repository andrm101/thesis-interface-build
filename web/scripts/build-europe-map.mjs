// Builds public/europe.json — country outlines for the dashboard maps.
//
//   node scripts/build-europe-map.mjs
//
// Source: Natural Earth 1:50m admin-0 countries (public domain) via the
// world-atlas package (TopoJSON). Keeps European countries only, drops
// overseas territories (French Guiana, Réunion, the Canaries, Svalbard…) by
// keeping only polygons whose centre lies inside the European frame, and
// rounds coordinates to 0.01° (about 1 km) to keep the file small.
import { readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { feature } from 'topojson-client';

const require = createRequire(import.meta.url);
const topo = JSON.parse(readFileSync(require.resolve('world-atlas/countries-50m.json'), 'utf8'));

const EUROPE = new Set([
  'Albania', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herz.', 'Bulgaria', 'Croatia',
  'Cyprus', 'N. Cyprus', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany',
  'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Kosovo', 'Latvia', 'Liechtenstein',
  'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 'Montenegro', 'Netherlands',
  'North Macedonia', 'Macedonia', 'Norway', 'Poland', 'Portugal', 'Romania', 'Serbia',
  'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'Ukraine',
  'United Kingdom', 'Andorra', 'Monaco', 'San Marino', 'Vatican',
]);
const FRAME = { lon: [-25, 45], lat: [34, 72] };

const round = c => Math.round(c * 100) / 100;
const centre = ring => {
  const n = ring.length;
  return [ring.reduce((s, p) => s + p[0], 0) / n, ring.reduce((s, p) => s + p[1], 0) / n];
};
const inFrame = ([x, y]) => x >= FRAME.lon[0] && x <= FRAME.lon[1] && y >= FRAME.lat[0] && y <= FRAME.lat[1];
const cleanRing = ring => {
  const out = [];
  for (const p of ring) {
    const q = [round(p[0]), round(p[1])];
    const last = out[out.length - 1];
    if (!last || last[0] !== q[0] || last[1] !== q[1]) out.push(q);
  }
  return out;
};

const countries = feature(topo, topo.objects.countries).features;
const features = [];
for (const f of countries) {
  const name = f.properties.name;
  if (!EUROPE.has(name) || !f.geometry) continue;
  const polys = f.geometry.type === 'Polygon' ? [f.geometry.coordinates] : f.geometry.coordinates;
  const kept = polys
    .filter(poly => inFrame(centre(poly[0])))
    .map(poly => poly.map(cleanRing).filter(r => r.length >= 4));
  if (!kept.length) continue;
  features.push({
    type: 'Feature',
    properties: { name: name === 'Macedonia' ? 'North Macedonia' : name },
    geometry: { type: 'MultiPolygon', coordinates: kept },
  });
}
features.sort((a, b) => a.properties.name.localeCompare(b.properties.name));
writeFileSync(new URL('../public/europe.json', import.meta.url),
  JSON.stringify({ type: 'FeatureCollection', features }));
console.log(`public/europe.json: ${features.length} countries`);
