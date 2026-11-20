import { useEffect, useState } from "react";

interface FunnelStats {
  total_listings: number;
  cleared_listings: number;
  evaluated_listings: number;
  triggered_llm_gate: number;
  confirmed_buys: number;
  rejected_traps: number;
  llm_calls_saved_pct: number;
  last_evaluated_at: string | null;
}

interface Deal {
  listing_id: number;
  source: string;
  url: string;
  title: string;
  seller_rating: number | null;
  canonical_sku: string;
  brand_tier: string;
  condition_score: number;
  has_warranty: boolean;
  has_original_box: boolean;
  is_mined_or_repaired: boolean;
  evaluated_price: number;
  q10_price: number;
  q50_price: number;
  expected_net_profit: number;
  discount_vs_q50_pct: number;
  audit_verdict: string | null;
  trap_flags: string[];
  arbitrage_rationale: string | null;
}

interface SkuBenchmark {
  canonical_sku: string;
  active_count: number;
  cleared_count: number;
  median_q10: number;
  median_q50: number;
  median_q90: number;
}

type Tab = "BUY" | "MANUAL_REVIEW" | "REJECT_TRAP" | "SKUS";

const VERDICT_STYLES: Record<string, string> = {
  BUY: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  MANUAL_REVIEW: "bg-amber-500/15 text-amber-400 border-amber-500/30",
  REJECT_TRAP: "bg-rose-500/15 text-rose-400 border-rose-500/30",
};

export default function App() {
  const [stats, setStats] = useState<FunnelStats | null>(null);
  const [deals, setDeals] = useState<Deal[]>([]);
  const [skus, setSkus] = useState<SkuBenchmark[]>([]);
  const [tab, setTab] = useState<Tab>("BUY");
  const [selectedSku, setSelectedSku] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch("/api/stats").then((r) => (r.ok ? r.json() : Promise.reject(r.statusText))),
      fetch("/api/skus").then((r) => (r.ok ? r.json() : Promise.reject(r.statusText))),
    ])
      .then(([s, k]) => {
        setStats(s);
        setSkus(k);
      })
      .catch((e) => setError(`API error: ${e}`));
  }, []);

  useEffect(() => {
    if (tab === "SKUS") return;
    setLoading(true);
    const params = new URLSearchParams({ verdict: tab, only_gated: "true", limit: "100" });
    if (selectedSku) params.set("sku", selectedSku);

    fetch(`/api/deals?${params.toString()}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)))
      .then((data) => {
        setDeals(data);
        setError(null);
      })
      .catch((e) => setError(`Failed to load deals: ${e}`))
      .finally(() => setLoading(false));
  }, [tab, selectedSku]);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 antialiased">
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {/* Header */}
        <header className="mb-8 flex flex-col justify-between gap-4 border-b border-zinc-800 pb-6 sm:flex-row sm:items-end">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight text-white">
              EU GPU Arbitrage Engine
            </h1>
            <p className="mt-1 text-sm text-zinc-400">
              LightGBM Quantile Valuation (Q10 / Q50) + Post-Gate LLM Trap Auditor across eBay DE, FR &amp; IT
            </p>
          </div>
          {stats?.last_evaluated_at && (
            <span className="text-xs text-zinc-500">
              Last run: {new Date(stats.last_evaluated_at).toLocaleString()}
            </span>
          )}
        </header>

        {/* Telemetry Strip */}
        {stats && (
          <section className="mb-8 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <StatCard label="Listings Scanned" value={stats.evaluated_listings.toLocaleString()} />
            <StatCard
              label="LLM Calls Saved"
              value={`${stats.llm_calls_saved_pct}%`}
              sub={`${stats.triggered_llm_gate} gated`}
            />
            <StatCard
              label="Confirmed Buys"
              value={stats.confirmed_buys.toLocaleString()}
              accent="text-emerald-400"
            />
            <StatCard
              label="Traps Blocked"
              value={stats.rejected_traps.toLocaleString()}
              accent="text-rose-400"
            />
          </section>
        )}

        {/* Controls */}
        <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
          <div className="flex gap-1 rounded-lg bg-zinc-900 p-1 border border-zinc-800">
            {(
              [
                ["BUY", "Approved Deals"],
                ["MANUAL_REVIEW", "Manual Review"],
                ["REJECT_TRAP", "Blocked Traps"],
                ["SKUS", "SKU Benchmarks"],
              ] as [Tab, string][]
            ).map(([key, label]) => (
              <button
                key={key}
                onClick={() => setTab(key)}
                className={`rounded-md px-3 py-1.5 text-sm font-medium transition ${
                  tab === key
                    ? "bg-zinc-800 text-white shadow"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {tab !== "SKUS" && (
            <select
              value={selectedSku}
              onChange={(e) => setSelectedSku(e.target.value)}
              className="rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-1.5 text-sm text-zinc-200 focus:border-zinc-600 focus:outline-none"
            >
              <option value="">All SKUs</option>
              {skus.map((s) => (
                <option key={s.canonical_sku} value={s.canonical_sku}>
                  {s.canonical_sku} ({s.active_count})
                </option>
              ))}
            </select>
          )}
        </div>

        {error && (
          <div className="mb-6 rounded-lg border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-300">
            {error}
          </div>
        )}

        {/* Main Content */}
        {tab === "SKUS" ? (
          <div className="overflow-x-auto rounded-xl border border-zinc-800 bg-zinc-900/50">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-zinc-800 text-xs uppercase text-zinc-400">
                <tr>
                  <th className="px-4 py-3">SKU</th>
                  <th className="px-4 py-3">Active</th>
                  <th className="px-4 py-3">Q10 Floor</th>
                  <th className="px-4 py-3">Q50 Median</th>
                  <th className="px-4 py-3">Q90 Ceiling</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/70">
                {skus.map((s) => (
                  <tr key={s.canonical_sku} className="hover:bg-zinc-900">
                    <td className="px-4 py-3 font-medium text-white">{s.canonical_sku}</td>
                    <td className="px-4 py-3 text-zinc-400">{s.active_count}</td>
                    <td className="px-4 py-3 text-emerald-400">€{s.median_q10.toFixed(0)}</td>
                    <td className="px-4 py-3 text-zinc-200">€{s.median_q50.toFixed(0)}</td>
                    <td className="px-4 py-3 text-zinc-400">€{s.median_q90.toFixed(0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : loading ? (
          <div className="py-16 text-center text-sm text-zinc-500">Loading deals...</div>
        ) : deals.length === 0 ? (
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 py-16 text-center text-sm text-zinc-500">
            No listings found for this filter.
          </div>
        ) : (
          <div className="space-y-3">
            {deals.map((d) => (
              <div
                key={d.listing_id}
                className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4 transition hover:border-zinc-700"
              >
                <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-start">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="rounded bg-zinc-800 px-2 py-0.5 text-xs font-semibold text-zinc-200">
                        {d.canonical_sku}
                      </span>
                      <span className="rounded bg-zinc-800/80 px-2 py-0.5 text-xs text-zinc-400">
                        {d.source.replace("ebay_", "").toUpperCase()}
                      </span>
                      {d.audit_verdict && (
                        <span
                          className={`rounded border px-2 py-0.5 text-xs font-medium ${
                            VERDICT_STYLES[d.audit_verdict] ?? "border-zinc-700 text-zinc-300"
                          }`}
                        >
                          {d.audit_verdict}
                        </span>
                      )}
                      {d.has_original_box && (
                        <span className="text-xs text-zinc-400">Boxed</span>
                      )}
                      {d.has_warranty && (
                        <span className="text-xs text-zinc-400">Warranty</span>
                      )}
                    </div>

                    <a
                      href={d.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="mt-2 block truncate text-base font-medium text-white hover:text-emerald-400"
                    >
                      {d.title}
                    </a>

                    {d.arbitrage_rationale && (
                      <p className="mt-1.5 text-sm text-zinc-400">{d.arbitrage_rationale}</p>
                    )}

                    {d.trap_flags.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {d.trap_flags.map((flag) => (
                          <span
                            key={flag}
                            className="rounded bg-rose-500/10 px-2 py-0.5 text-xs font-mono text-rose-300"
                          >
                            {flag}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Pricing Column */}
                  <div className="flex shrink-0 flex-row items-end justify-between gap-4 border-t border-zinc-800 pt-3 sm:flex-col sm:border-t-0 sm:pt-0 sm:text-right">
                    <div>
                      <div className="text-xl font-semibold text-white">
                        €{d.evaluated_price.toFixed(0)}
                      </div>
                      <div className="text-xs text-zinc-400">
                        Fair Q50: €{d.q50_price.toFixed(0)} (-{d.discount_vs_q50_pct}%)
                      </div>
                    </div>
                    <div className="text-sm font-medium text-emerald-400">
                      Est. Net: €{d.expected_net_profit.toFixed(0)}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({
  label,
  value,
  sub,
  accent = "text-white",
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: string;
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
      <div className="text-xs text-zinc-400">{label}</div>
      <div className={`mt-1 text-2xl font-semibold ${accent}`}>{value}</div>
      {sub && <div className="mt-0.5 text-xs text-zinc-500">{sub}</div>}
    </div>
  );
}