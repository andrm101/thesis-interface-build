import { Component, ElementRef, inject, signal, viewChild } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { marked } from 'marked';
import { ApiService } from '../core/api.service';
import { AssistantReply } from '../core/models';

interface Turn { question: string; reply?: AssistantReply; html?: string; error?: string; }

const SUGGESTIONS = [
  'Does public R&D funding crowd in private R&D, and does it differ by group?',
  'Did R&D tax-incentive reforms raise R&D intensity? Are the pre-trends OK?',
  'How has Poland\'s R&D intensity changed since 2000 compared with the Innovative group?',
  'What happens to output per worker after a cyclically adjusted fiscal tightening?',
];

@Component({
  selector: 'app-assistant',
  imports: [FormsModule],
  template: `
  <section class="page">
    <header class="page-head">
      <h1>Research assistant</h1>
      <p>Ask in plain language. Claude answers by running the thesis's own analyses — the
         same functions as the other pages — and every number in the answer is checked
         against the results it came from.</p>
    </header>

    @if (status(); as st) {
      @if (!st.available) {
        <div class="card setup">
          <h2>The assistant is not enabled on this server</h2>
          <p class="note">{{ st.detail }}.</p>
          <p>To enable it, install the optional packages and start the API with a Claude API key:</p>
          <pre>pip install -r requirements-llm.txt
export ANTHROPIC_API_KEY=sk-ant-…        # Windows: set ANTHROPIC_API_KEY=…
uvicorn api.main:app --port 8000

# or with Docker
ANTHROPIC_API_KEY=sk-ant-… docker compose up --build</pre>
        </div>
      }
    }

    <div class="thread">
      @for (t of turns(); track $index) {
        <div class="q"><span>{{ t.question }}</span></div>
        <div class="card a">
          @if (t.reply) {
            <div class="md" [innerHTML]="t.html"></div>
            <div class="meta">
              @if (t.reply.grounding.unverified.length) {
                <span class="badge bad" title="These numbers do not appear in any tool result">
                  ⚠ {{ t.reply.grounding.unverified.length }} unverified: {{ t.reply.grounding.unverified.join(', ') }}</span>
              } @else if (t.reply.grounding.verified.length) {
                <span class="badge good">✓ all {{ t.reply.grounding.verified.length }} numbers traced to results</span>
              }
              @for (c of t.reply.tool_calls; track c.result_id) {
                <details class="call">
                  <summary><b>{{ c.result_id }}</b> {{ c.name }}@if (c.error) { <span class="err"> · failed</span> }</summary>
                  <code>{{ args(c.input) }}</code>
                  @if (c.error) { <p class="error">{{ c.error }}</p> }
                </details>
              }
            </div>
          } @else if (t.error) {
            <p class="error">{{ t.error }}</p>
          } @else {
            <p class="working"><span class="spinner"></span> Running analyses… this can take up to a minute.</p>
          }
        </div>
      }
    </div>

    <div class="card composer">
      @if (!turns().length) {
        <div class="chips">
          @for (s of suggestions; track s) {
            <button type="button" class="chip" (click)="question = s; ask()" [disabled]="!ready()">{{ s }}</button>
          }
        </div>
      }
      <div class="row">
        <textarea #box rows="2" [(ngModel)]="question" placeholder="Ask about R&D, productivity, convergence or policy…"
                  (keydown.enter)="$event.preventDefault(); ask()" [disabled]="!ready()"></textarea>
        <div class="btns">
          <button class="btn primary" type="button" (click)="ask()" [disabled]="!ready() || busy() || !question.trim()">Ask</button>
          <button class="btn" type="button" (click)="reset()" [disabled]="busy() || !turns().length">New conversation</button>
        </div>
      </div>
      <p class="note">Model: {{ status()?.model ?? '…' }}. Answers can be wrong; check the cited results
        (r1, r2, …) before quoting a number.</p>
    </div>
  </section>`,
  styles: [`
    .setup pre { background: var(--surface-2); padding: 12px; border-radius: 8px; font-size: 12.5px; overflow-x: auto; }
    .setup p { margin: 8px 0; }
    .thread { display: grid; gap: 12px; }
    .q { display: flex; justify-content: flex-end; }
    .q span { background: var(--accent); color: var(--accent-ink); padding: 8px 14px; border-radius: 14px 14px 4px 14px;
      max-width: 75%; white-space: pre-wrap; }
    .a .meta { display: flex; flex-wrap: wrap; gap: 6px; align-items: flex-start; margin-top: 10px; }
    .call { font-size: 12px; border: 1px solid var(--border); border-radius: 8px; padding: 2px 8px; background: var(--surface-2); }
    .call summary { cursor: pointer; color: var(--ink-2); }
    .call code { display: block; white-space: pre-wrap; font-size: 11.5px; color: var(--ink-2); padding: 4px 0; max-width: 520px; }
    .err { color: var(--critical); }
    .working { color: var(--ink-2); }
    .composer .row { display: flex; gap: 10px; align-items: stretch; margin-top: 10px; }
    .composer .chips { margin-bottom: 4px; }
    textarea { flex: 1; font: inherit; color: var(--ink); background: var(--surface-1); border: 1px solid var(--axis);
      border-radius: 8px; padding: 8px 10px; resize: vertical; min-width: 0; }
    .btns { display: grid; gap: 6px; align-content: start; }
    :host ::ng-deep .cite { font-size: 11px; padding: 0 5px; border-radius: 4px; background: var(--surface-2);
      color: var(--accent); font-weight: 600; }
    @media (max-width: 700px) { .composer .row { flex-direction: column; } .q span { max-width: 90%; } }
  `],
})
export class AssistantPage {
  private api = inject(ApiService);
  protected suggestions = SUGGESTIONS;
  protected status = toSignal(this.api.assistantStatus(), { initialValue: undefined });
  protected turns = signal<Turn[]>([]);
  protected busy = signal(false);
  protected question = '';
  private conversation: string | null = null;
  private box = viewChild<ElementRef<HTMLTextAreaElement>>('box');

  protected ready() { return !!this.status()?.available; }

  protected args(input: Record<string, unknown>) {
    return Object.entries(input).map(([k, v]) => `${k} = ${JSON.stringify(v)}`).join('\n') || '(no arguments)';
  }

  protected ask() {
    const q = this.question.trim();
    if (!q || this.busy() || !this.ready()) return;
    this.question = '';
    this.busy.set(true);
    const turn: Turn = { question: q };
    this.turns.update(t => [...t, turn]);
    this.api.ask(q, this.conversation).subscribe({
      next: r => {
        this.conversation = r.conversation_id;
        const md = marked.parse(r.answer, { async: false }) as string;
        // citations [r1] → small badges
        const html = md.replace(/\[(r\d+)\]/g, '<span class="cite">$1</span>');
        this.patch(turn, { reply: r, html });
      },
      error: (e: Error) => this.patch(turn, { error: e.message }),
    });
  }

  private patch(turn: Turn, upd: Partial<Turn>) {
    this.turns.update(ts => ts.map(t => (t === turn ? { ...t, ...upd } : t)));
    this.busy.set(false);
    queueMicrotask(() => this.box()?.nativeElement.focus());
  }

  protected reset() {
    this.conversation = null;
    this.turns.set([]);
  }
}
