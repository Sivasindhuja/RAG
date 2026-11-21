import React, { useMemo, useState } from "react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

async function queryRag(question, mode = "extractive") {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, mode }),
  });
  let data = null;
  try {
    data = await res.json();
  } catch {
    data = null;
  }
  if (!res.ok) {
    const detail = data?.detail || data?.error || `HTTP ${res.status}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function StatusBadge({ status, abstained }) {
  const label = abstained ? "abstained" : status || "unknown";
  const cls =
    label === "answered"
      ? "bg-green-100 text-green-800 border-green-200 dark:bg-green-950 dark:text-green-300 dark:border-green-900"
      : label === "abstained"
      ? "bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-950 dark:text-amber-300 dark:border-amber-900"
      : "bg-red-100 text-red-800 border-red-200 dark:bg-red-950 dark:text-red-300 dark:border-red-900";
  return (
    <span className={`inline-flex items-center rounded border px-2 py-0.5 text-xs font-medium ${cls}`}>
      {label}
    </span>
  );
}

function CitationCard({ citation, index }) {
  return (
    <div className="rounded-md border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-3 text-sm">
      <div className="flex flex-wrap items-center gap-2 mb-1">
        <span className="font-semibold text-orange-700 dark:text-orange-400">{citation.marker || `[${index + 1}]`}</span>
        <span className="text-slate-700 dark:text-slate-200">
          {citation.section ? `Section ${citation.section}` : "Section n/a"}
          {citation.heading ? ` — ${citation.heading}` : ""}
        </span>
        <span className="text-xs text-slate-500">
          p. {citation.start_page}{citation.end_page !== citation.start_page ? `–${citation.end_page}` : ""}
        </span>
      </div>
      <div className="text-xs text-slate-500 dark:text-slate-400 mb-2 break-all">
        <div><span className="font-medium">chunk:</span> {citation.chunk_id}</div>
        <div><span className="font-medium">source:</span> {citation.source} | <span className="font-medium">offsets:</span> {citation.start_offset}–{citation.end_offset}</div>
        {citation.article ? <div><span className="font-medium">article:</span> {citation.article}</div> : null}
        {typeof citation.distance === "number" ? <div><span className="font-medium">distance:</span> {citation.distance.toFixed(4)}</div> : null}
      </div>
      {citation.quote ? (
        <pre className="whitespace-pre-wrap text-xs bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded p-2 text-slate-700 dark:text-slate-300">
          {citation.quote}
        </pre>
      ) : null}
    </div>
  );
}

function ResultPanel({ result }) {
  if (!result) return null;
  const citations = result.citations || [];
  const statements = result.statements || [];
  const retrieved = result.retrieved || [];
  const errors = [...(result.errors || []), ...(result.generation?.errors || [])].filter(Boolean);
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <StatusBadge status={result.status} abstained={result.abstained} />
        <span className="text-xs text-slate-500">mode: {result.mode}</span>
        {result.request_id ? <span className="text-xs text-slate-500">request: {result.request_id}</span> : null}
      </div>

      <div className="rounded-md border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-4">
        <div className="text-sm font-semibold text-slate-900 dark:text-slate-100 mb-2">Answer</div>
        <div className="whitespace-pre-wrap text-sm text-slate-800 dark:text-slate-200">{result.answer || "(no answer)"}</div>
      </div>

      {result.abstained ? (
        <Alert className="bg-amber-50 dark:bg-amber-950/30 border-amber-200 dark:border-amber-900">
          <AlertTitle className="text-amber-800 dark:text-amber-300">Abstained (insufficient/unsupported evidence)</AlertTitle>
          <AlertDescription className="text-amber-900/90 dark:text-amber-200 text-sm">
            {result.abstention_reason || "No sufficient supporting evidence was selected."}
          </AlertDescription>
        </Alert>
      ) : null}

      {errors.length ? (
        <Alert className="bg-red-50 dark:bg-red-950/30 border-red-200 dark:border-red-900">
          <AlertTitle className="text-red-800 dark:text-red-300">Warnings / errors</AlertTitle>
          <AlertDescription className="text-red-900/90 dark:text-red-200 text-xs whitespace-pre-wrap">
            {errors.join("\n")}
          </AlertDescription>
        </Alert>
      ) : null}

      <div>
        <div className="text-sm font-semibold text-slate-900 dark:text-slate-100 mb-2">Citations (app-resolved, inspectable)</div>
        {citations.length ? (
          <div className="grid grid-cols-1 gap-3">
            {citations.map((c, i) => <CitationCard key={`${c.chunk_id}-${c.marker}-${i}`} citation={c} index={i} />)}
          </div>
        ) : (
          <div className="text-sm text-slate-500">No citations.</div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <details className="rounded-md border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-3">
          <summary className="cursor-pointer text-sm font-semibold text-slate-900 dark:text-slate-100">Selected evidence statements ({statements.length})</summary>
          <div className="mt-2 space-y-2">
            {statements.length ? statements.map((s, i) => (
              <div key={i} className="text-xs border border-slate-200 dark:border-slate-800 rounded p-2">
                <div className="font-medium text-slate-700 dark:text-slate-200">{s.marker} · {s.chunk_id} · {s.start_offset}–{s.end_offset}</div>
                <div className="whitespace-pre-wrap text-slate-600 dark:text-slate-300 mt-1">{s.text}</div>
              </div>
            )) : <div className="text-xs text-slate-500">None.</div>}
          </div>
        </details>
        <details className="rounded-md border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-3">
          <summary className="cursor-pointer text-sm font-semibold text-slate-900 dark:text-slate-100">Retrieved chunks & diagnostics</summary>
          <div className="mt-2 space-y-2 text-xs text-slate-600 dark:text-slate-300">
            <div><span className="font-medium">Selected IDs:</span> {(result.selected_evidence_ids || []).join(", ") || "none"}</div>
            <div><span className="font-medium">Retrieved:</span></div>
            {retrieved.length ? retrieved.map((r, i) => (
              <div key={i} className="border border-slate-200 dark:border-slate-800 rounded p-2">
                #{r.rank} {r.chunk_id} {r.section ? `(sec ${r.section})` : ""} p.{r.start_page}{r.end_page !== r.start_page ? `–${r.end_page}` : ""} [{r.start_offset}–{r.end_offset}] {typeof r.distance === "number" ? `dist ${r.distance.toFixed(4)}` : ""}
              </div>
            )) : <div>None.</div>}
            <div><span className="font-medium">Timings (ms):</span> {JSON.stringify(result.timings_ms || {})}</div>
            {result.debug ? <pre className="whitespace-pre-wrap bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded p-2">{JSON.stringify({ selected_lexical_support: result.debug.selected_lexical_support, token_support: result.debug.token_support, best_distance: result.debug.best_distance, answerability_guard_applied: result.debug.answerability_guard_applied, answerability_guard_reason: result.debug.answerability_guard_reason }, null, 2)}</pre> : null}
          </div>
        </details>
      </div>
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState("chat");
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState("extractive");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [health, setHealth] = useState(null);

  const latest = history[0] || null;

  async function onAsk(selectedMode = mode) {
    const q = question.trim();
    const m = selectedMode || "extractive";
    if (!q || loading) return;
    setLoading(true);
    setError(null);
    try {
      const data = await queryRag(q, m);
      setHistory((prev) => [{ id: data.request_id || String(Date.now()), question: q, mode: m, result: data, at: new Date().toISOString() }, ...prev]);
      setQuestion("");
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      const data = await res.json();
      setHealth({ ok: res.ok, data });
    } catch (e) {
      setHealth({ ok: false, data: { error: e.message || String(e) } });
    }
  }

  return (
    <div className="min-h-screen bg-white dark:bg-slate-950 text-slate-900 dark:text-slate-50 flex flex-col items-center p-6 font-sans">
      <div className="w-full max-w-6xl space-y-6">
        <header className="border-b border-slate-200 dark:border-slate-800 pb-4 flex justify-between items-center gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-orange-500 to-amber-600">
              ISRO SATCOM NGP RAG
            </h1>
            <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
              Grounded Q&amp;A over <code>data/satcom-ngp.pdf</code> (extractive default, EXP-003, app-validated citations, explicit abstention).
            </p>
          </div>
          <div className="text-xs text-slate-500 text-right">
            <div>API: {API_BASE}</div>
            <Button variant="outline" size="sm" className="mt-1" onClick={checkHealth}>Check health</Button>
          </div>
        </header>

        {health ? (
          <Alert className={health.ok ? "bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-800" : "bg-red-50 dark:bg-red-950/30 border-red-200 dark:border-red-900"}>
            <AlertTitle className={health.ok ? "text-slate-800 dark:text-slate-200" : "text-red-800 dark:text-red-300"}>Health</AlertTitle>
            <AlertDescription className="text-xs whitespace-pre-wrap text-slate-600 dark:text-slate-300">{JSON.stringify(health.data, null, 2)}</AlertDescription>
          </Alert>
        ) : null}

        <div className="flex gap-2 flex-wrap">
          <Button variant={activeTab === "chat" ? "default" : "outline"} onClick={() => setActiveTab("chat")} className={activeTab === "chat" ? "bg-orange-600 hover:bg-orange-700 text-white" : ""}>Query</Button>
          <Button variant={activeTab === "guide" ? "default" : "outline"} onClick={() => setActiveTab("guide")} className={activeTab === "guide" ? "bg-orange-600 hover:bg-orange-700 text-white" : ""}>Guide</Button>
        </div>

        {activeTab === "chat" ? (
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
            <Card className="lg:col-span-2 bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-800">
              <CardHeader>
                <CardTitle className="text-slate-900 dark:text-slate-100">Ask a question</CardTitle>
                <CardDescription className="text-slate-500 dark:text-slate-400">
                  Answers are grounded in selected evidence with inspectable citations. The system abstains when evidence is insufficient.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <Input
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="e.g. What satellite communication services are classified as domestic services?"
                  onKeyDown={(e) => e.key === "Enter" && onAsk()}
                  disabled={loading}
                  className="bg-white dark:bg-slate-950 border-slate-200 dark:border-slate-800"
                />
                <div className="flex flex-wrap gap-2 items-center">
                  <label className="text-xs text-slate-600 dark:text-slate-300" htmlFor="mode">Mode</label>
                  <select
                    id="mode"
                    value={mode}
                    onChange={(e) => setMode(e.target.value)}
                    disabled={loading}
                    className="h-9 rounded-md border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 px-2 text-sm"
                  >
                    <option value="extractive">extractive (default)</option>
                    <option value="gemini">gemini (grounded, provider/model configured)</option>
                  </select>
                </div>
                <div className="flex gap-2 flex-wrap">
                  <Button onClick={() => onAsk()} disabled={loading || !question.trim()} className="bg-orange-600 hover:bg-orange-700 text-white">
                    {loading ? "Querying..." : `Ask (${mode})`}
                  </Button>
                  <Button variant="outline" onClick={() => { setQuestion("Who is responsible for obtaining licenses to offer services in a territory?"); }} disabled={loading}>Sample: Q05</Button>
                  <Button variant="outline" onClick={() => { setQuestion("What satellite communication services are classified as domestic services?"); }} disabled={loading}>Sample: Q12</Button>
                </div>
                {error ? (
                  <Alert className="bg-red-50 dark:bg-red-950/30 border-red-200 dark:border-red-900">
                    <AlertTitle className="text-red-800 dark:text-red-300">Request failed</AlertTitle>
                    <AlertDescription className="text-red-900/90 dark:text-red-200 text-sm whitespace-pre-wrap">{error}</AlertDescription>
                  </Alert>
                ) : null}
                <div className="text-xs text-slate-500">
                  Gemini generation uses the configured provider/model path (`google_generativeai` / `gemini-flash-lite-latest`) with evidence-only grounding, app-side citation validation, and safe fallback/abstention. Quality is reported from measured evaluations; not claimed production-accepted.
                </div>
                <div>
                  <div className="text-sm font-semibold mb-2">History</div>
                  <ScrollArea className="h-[220px] rounded border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-2">
                    <div className="space-y-2">
                      {history.length ? history.map((h) => (
                        <button key={h.id} className="w-full text-left rounded border border-slate-200 dark:border-slate-800 p-2 hover:bg-slate-50 dark:hover:bg-slate-900" onClick={() => setHistory((prev) => [h, ...prev.filter(x => x.id !== h.id)])}>
                          <div className="text-xs text-slate-500">{new Date(h.at).toLocaleTimeString()} · {h.result?.status}</div>
                          <div className="text-sm line-clamp-2">{h.question}</div>
                        </button>
                      )) : <div className="text-xs text-slate-500">No queries yet.</div>}
                    </div>
                  </ScrollArea>
                </div>
              </CardContent>
            </Card>

            <Card className="lg:col-span-3 bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-800">
              <CardHeader>
                <CardTitle className="text-slate-900 dark:text-slate-100">Result</CardTitle>
                <CardDescription className="text-slate-500 dark:text-slate-400">Inspectable evidence, citations, abstention and diagnostics.</CardDescription>
              </CardHeader>
              <CardContent>
                {loading && !latest ? <div className="text-sm text-slate-500 animate-pulse">Running retrieval, selection and citation resolution...</div> : null}
                {latest ? <ResultPanel result={latest.result} /> : (!loading ? <div className="text-sm text-slate-500">Ask a question to see a grounded response.</div> : null)}
              </CardContent>
            </Card>
          </div>
        ) : (
          <Card className="bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-800">
            <CardHeader>
              <CardTitle className="text-slate-900 dark:text-slate-100">Run & integration guide</CardTitle>
              <CardDescription className="text-slate-500 dark:text-slate-400">WP2 frontend connected to validated WP1 backend (`/query`).</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 text-sm text-slate-700 dark:text-slate-300">
              <div>
                <div className="font-semibold mb-1">1. Start backend (repo root)</div>
                <pre className="bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded p-2 text-xs overflow-x-auto">python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000</pre>
              </div>
              <div>
                <div className="font-semibold mb-1">2. Start frontend</div>
                <pre className="bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded p-2 text-xs overflow-x-auto">cd frontend
npm install
npm run dev</pre>
                <div className="text-xs text-slate-500 mt-1">Optional: set <code>VITE_API_BASE_URL</code> (default <code>http://localhost:8000</code>).</div>
              </div>
              <div>
                <div className="font-semibold mb-1">What this UI shows</div>
                <ul className="list-disc pl-5 text-xs text-slate-600 dark:text-slate-400 space-y-1">
                  <li>Grounded answer modes: extractive (default) and Gemini (configured provider/model path, evidence-only, app-validated citations, safe fallback/abstention).</li>
                  <li>App-resolved, inspectable citations: marker, section/heading, page(s), chunk ID, source offsets, quote.</li>
                  <li>Selected evidence statements, retrieved chunks, request ID/timings and guard diagnostics.</li>
                  <li>Known limitations remain where not yet fully verified (e.g. live public deployment/hosting constraints, external observability credentials, RAGAS environment compatibility, small draft CI set).</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
