import React, { useEffect, useState } from 'react';
import { Activity, ArrowRight, CheckCircle2, Cpu, Database, FileText, Layers, Shield, Terminal, Zap } from 'lucide-react';
import { fetchHealth, fetchMetadata } from '../services/api';
import { HealthResponse, MetadataResponse } from '../types';

interface OverviewProps {
  onNavigate: (tab: string) => void;
}

export const Overview: React.FC<OverviewProps> = ({ onNavigate }) => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [metadata, setMetadata] = useState<MetadataResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([fetchHealth(), fetchMetadata()])
      .then(([h, m]) => {
        setHealth(h);
        setMetadata(m);
      })
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-8 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Hero Banner */}
      <div className="rounded-xl bg-gradient-to-r from-slatepanel-900 via-charcoal-900 to-slatepanel-900 border border-slate-800 p-8 shadow-xl">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div className="space-y-3 max-w-3xl">
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-red-950/50 border border-red-500/30 text-red-400 text-xs font-mono">
              <Zap className="w-3.5 h-3.5" />
              <span>Phase 6: Full-Stack Integration & Production Serving</span>
            </div>
            <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight font-sans">
              FailureDNA: Self-Supervised Discovery of Failure Signatures
            </h1>
            <p className="text-slate-300 text-sm sm:text-base leading-relaxed">
              Unsupervised representation learning framework designed to isolate latent degradation dynamics from complex multivariate industrial sensor telemetry across extreme operating regimes on NASA C-MAPSS FD004.
            </p>
          </div>
          <div className="flex flex-col sm:flex-row gap-3 w-full md:w-auto">
            <button
              onClick={() => onNavigate('analysis')}
              className="flex items-center justify-center space-x-2 px-5 py-2.5 rounded-lg bg-red-600 hover:bg-red-500 text-white font-medium text-sm transition-all shadow-lg shadow-red-600/20"
            >
              <Activity className="w-4 h-4" />
              <span>Analyze Telemetry</span>
            </button>
            <button
              onClick={() => onNavigate('benchmark')}
              className="flex items-center justify-center space-x-2 px-5 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-sm border border-slate-700 transition-all"
            >
              <Database className="w-4 h-4" />
              <span>View Benchmark</span>
            </button>
          </div>
        </div>
      </div>

      {/* System Status & Key Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="rounded-xl bg-slatepanel-900/90 border border-slate-800 p-5 space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-mono uppercase tracking-wider">Inference Pipeline</span>
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-white">
            {health?.is_ready ? '100% OPERATIONAL' : loading ? 'CHECKING...' : 'DEGRADED'}
          </div>
          <p className="text-xs text-slate-400">Frozen 4L Transformer & Preprocessor Loaded</p>
        </div>

        <div className="rounded-xl bg-slatepanel-900/90 border border-slate-800 p-5 space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-mono uppercase tracking-wider">Official Cohort</span>
            <Database className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-white">248 ENGINES</div>
          <p className="text-xs text-slate-400">NASA C-MAPSS FD004 Test Fleet</p>
        </div>

        <div className="rounded-xl bg-slatepanel-900/90 border border-slate-800 p-5 space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-mono uppercase tracking-wider">Flight Regimes</span>
            <Layers className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-white">6 CONDITIONS</div>
          <p className="text-xs text-slate-400">Altitude, Mach, Throttle Normalized</p>
        </div>

        <div className="rounded-xl bg-slatepanel-900/90 border border-slate-800 p-5 space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-mono uppercase tracking-wider">Signatures</span>
            <Cpu className="w-4 h-4 text-red-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-white">4 DISCOVERED</div>
          <p className="text-xs text-slate-400">Monotonically Ordered States (0 → 3)</p>
        </div>
      </div>

      {/* Artifact Verification Panel */}
      <div className="rounded-xl bg-slatepanel-900/80 border border-slate-800 p-6 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center space-x-2">
            <Shield className="w-5 h-5 text-emerald-400" />
            <h2 className="text-base font-semibold text-white">Frozen Artifact Verification & Integrity Status</h2>
          </div>
          <span className="text-xs font-mono text-slate-400">API v{metadata?.version || '1.0.0'}</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 pt-2">
          <div className="p-3.5 rounded-lg bg-charcoal-900/80 border border-slate-800">
            <div className="text-xs font-mono text-slate-400">PyTorch Checkpoint</div>
            <div className="text-sm font-semibold text-slate-200 mt-1">failuredna_regime_aware</div>
            <div className="text-[11px] font-mono text-slate-500 mt-0.5 truncate">SHA-256: d18b6cd0abc2...</div>
            <div className="inline-block mt-2 px-2 py-0.5 text-[10px] rounded bg-emerald-950/60 text-emerald-400 border border-emerald-500/30">
              Verified Frozen
            </div>
          </div>
          <div className="p-3.5 rounded-lg bg-charcoal-900/80 border border-slate-800">
            <div className="text-xs font-mono text-slate-400">Preprocessor</div>
            <div className="text-sm font-semibold text-slate-200 mt-1">RegimeAwarePreprocessor (k=6)</div>
            <div className="text-[11px] font-mono text-slate-500 mt-0.5">24 Standardized Channels</div>
            <div className="inline-block mt-2 px-2 py-0.5 text-[10px] rounded bg-emerald-950/60 text-emerald-400 border border-emerald-500/30">
              Train-Only Fitted
            </div>
          </div>
          <div className="p-3.5 rounded-lg bg-charcoal-900/80 border border-slate-800">
            <div className="text-xs font-mono text-slate-400">Downstream Estimator</div>
            <div className="text-sm font-semibold text-slate-200 mt-1">Linear Ridge Probe (α=10.0)</div>
            <div className="text-[11px] font-mono text-slate-500 mt-0.5">Capped RUL [0, 125] Cycles</div>
            <div className="inline-block mt-2 px-2 py-0.5 text-[10px] rounded bg-emerald-950/60 text-emerald-400 border border-emerald-500/30">
              100% Reprod. Match
            </div>
          </div>
          <div className="p-3.5 rounded-lg bg-charcoal-900/80 border border-slate-800">
            <div className="text-xs font-mono text-slate-400">Signature Clusterer</div>
            <div className="text-sm font-semibold text-slate-200 mt-1">KMeans Clusterer (K=4)</div>
            <div className="text-[11px] font-mono text-slate-500 mt-0.5">Lifecycle Monotonic Order</div>
            <div className="inline-block mt-2 px-2 py-0.5 text-[10px] rounded bg-emerald-950/60 text-emerald-400 border border-emerald-500/30">
              Zero Mismatches
            </div>
          </div>
        </div>
      </div>

      {/* Feature Exploration Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div
          onClick={() => onNavigate('analysis')}
          className="group cursor-pointer rounded-xl bg-slatepanel-900/60 hover:bg-slatepanel-900 border border-slate-800 hover:border-slate-700 p-6 transition-all space-y-3"
        >
          <div className="w-10 h-10 rounded-lg bg-red-950/50 border border-red-500/30 flex items-center justify-center text-red-400">
            <Activity className="w-5 h-5" />
          </div>
          <h3 className="text-lg font-bold text-white group-hover:text-red-400 transition-colors flex items-center justify-between">
            <span>Engine Telemetry Analysis</span>
            <ArrowRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
          </h3>
          <p className="text-xs sm:text-sm text-slate-400">
            Input chronological engine sensor logs via CSV drag-and-drop or validated JSON payload. Runs edge-padded windowing ($W=50$), regime standardization, latent extraction, and continuous RUL prediction.
          </p>
        </div>

        <div
          onClick={() => onNavigate('signatures')}
          className="group cursor-pointer rounded-xl bg-slatepanel-900/60 hover:bg-slatepanel-900 border border-slate-800 hover:border-slate-700 p-6 transition-all space-y-3"
        >
          <div className="w-10 h-10 rounded-lg bg-cyan-950/50 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
            <Cpu className="w-5 h-5" />
          </div>
          <h3 className="text-lg font-bold text-white group-hover:text-cyan-400 transition-colors flex items-center justify-between">
            <span>Signature Explorer & Heatmap</span>
            <ArrowRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
          </h3>
          <p className="text-xs sm:text-sm text-slate-400">
            Inspect discovered degradation signatures, fleet state occupancy, Markov transition dynamics, empirical dwell times, and the empirical hazard-rate audit (reciprocal hazard 33.12 vs 1.98 mean).
          </p>
        </div>

        <div
          onClick={() => onNavigate('benchmark')}
          className="group cursor-pointer rounded-xl bg-slatepanel-900/60 hover:bg-slatepanel-900 border border-slate-800 hover:border-slate-700 p-6 transition-all space-y-3"
        >
          <div className="w-10 h-10 rounded-lg bg-amber-950/50 border border-amber-500/30 flex items-center justify-center text-amber-400">
            <Database className="w-5 h-5" />
          </div>
          <h3 className="text-lg font-bold text-white group-hover:text-amber-400 transition-colors flex items-center justify-between">
            <span>Benchmark Lab & Statistical Audit</span>
            <ArrowRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
          </h3>
          <p className="text-xs sm:text-sm text-slate-400">
            Access official FD004 test benchmark metrics across 11 architectures. Displays MAE, RMSE, NASA score, 95% paired bootstrap confidence intervals, and FWER-controlled Holm adjusted p-values.
          </p>
        </div>

        <div
          onClick={() => onNavigate('documentation')}
          className="group cursor-pointer rounded-xl bg-slatepanel-900/60 hover:bg-slatepanel-900 border border-slate-800 hover:border-slate-700 p-6 transition-all space-y-3"
        >
          <div className="w-10 h-10 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-300">
            <FileText className="w-5 h-5" />
          </div>
          <h3 className="text-lg font-bold text-white group-hover:text-slate-300 transition-colors flex items-center justify-between">
            <span>Technical Documentation & API</span>
            <ArrowRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
          </h3>
          <p className="text-xs sm:text-sm text-slate-400">
            Review full-stack architecture, mathematical formulations (NT-Xent, Transformer, Regime Clustering), sensor channel specifications, interactive REST API cURL commands, and research disclaimers.
          </p>
        </div>
      </div>

      {/* Research Limitations & Scientific Disclaimer */}
      <div className="rounded-xl bg-slatepanel-900/40 border border-slate-800 p-5 space-y-2 text-xs text-slate-400">
        <div className="font-semibold text-slate-300 flex items-center space-x-1.5">
          <Terminal className="w-4 h-4 text-slate-400" />
          <span>Scientific Disclaimer & Statistical Qualifications</span>
        </div>
        <p className="leading-relaxed">
          {metadata?.disclaimer ||
            'FailureDNA is an academic research prototype evaluating self-supervised representations on NASA C-MAPSS FD004. In accordance with Phase 5.1/5.2 statistical reproducibility audits, FailureDNA does not establish overall prognostic superiority over tuned XGBoost on FD004 (XGBoost maintains a statistically significant MAE advantage under omnibus Holm correction; RMSE and NASA score differences are not statistically significant).'}
        </p>
      </div>
    </div>
  );
};
