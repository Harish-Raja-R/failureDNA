import React, { useEffect, useState } from 'react';
import { BarChart3, Cpu, Info, RefreshCw, ShieldAlert } from 'lucide-react';
import { fetchSignatures } from '../services/api';
import { SignaturesResponse } from '../types';
import { SignatureBadge } from '../components/StatusBadge';

export const SignatureExplorer: React.FC = () => {
  const [data, setData] = useState<SignaturesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchSignatures();
      setData(res);
    } catch (err: any) {
      setError(err.message || 'Failed to load signature data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const stateDetails = [
    {
      state: 0,
      title: 'State 0: Baseline Healthy / Nominal',
      occupancy: '36.3%',
      dwell: '28.4 cycles',
      description: 'Initial operational phase with balanced thermal and pressure ratios across low and high pressure turbines.',
      color: 'border-emerald-500/40 text-emerald-400 bg-emerald-950/20',
    },
    {
      state: 1,
      title: 'State 1: Early Degradation Onset',
      occupancy: '36.3%',
      dwell: '28.6 cycles',
      description: 'Subtle efficiency decline in compressor and turbine stages, initiating gradual exhaust gas temperature elevation.',
      color: 'border-cyan-500/40 text-cyan-400 bg-cyan-950/20',
    },
    {
      state: 2,
      title: 'State 2: Accelerated Wear Stage',
      occupancy: '7.8%',
      dwell: '12.9 cycles',
      description: 'Pronounced aerodynamic flow resistance, increased fuel-to-pressure ratio (phi), and heightened mechanical stress.',
      color: 'border-amber-500/40 text-amber-400 bg-amber-950/20',
    },
    {
      state: 3,
      title: 'State 3: Critical Terminal State',
      occupancy: '6.8%',
      dwell: '15.0 cycles',
      description: 'Severe cumulative degradation preceding functional failure. High transition probability to irreversible engine shutdown.',
      color: 'border-red-500/40 text-red-400 bg-red-950/20',
    },
  ];

  return (
    <div className="space-y-8 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white font-sans flex items-center space-x-3">
            <Cpu className="w-7 h-7 text-red-500" />
            <span>Unsupervised Failure Signature Discovery</span>
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Latent representation clustering, Markov state transition dynamics, and empirical hazard rate audits on NASA C-MAPSS FD004.
          </p>
        </div>
        <button
          onClick={loadData}
          disabled={loading}
          className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-all"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh Artifact Data</span>
        </button>
      </div>

      {error && (
        <div className="rounded-lg bg-red-950/60 border border-red-500/40 p-4 text-red-300 text-xs">
          Failed loading signature artifacts: {error}
        </div>
      )}

      {/* Discovered Signatures Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {stateDetails.map((item) => (
          <div key={item.state} className={`rounded-xl border p-5 space-y-3 ${item.color}`}>
            <div className="flex items-center justify-between">
              <SignatureBadge state={item.state} size="sm" />
              <span className="text-xs font-mono font-bold text-slate-200">{item.occupancy} Fleet</span>
            </div>
            <h3 className="text-sm font-bold text-white">{item.title}</h3>
            <p className="text-xs text-slate-300 leading-relaxed">{item.description}</p>
            <div className="text-[11px] font-mono text-slate-400 pt-2 border-t border-slate-800/80 flex justify-between">
              <span>Empirical Mean Dwell:</span>
              <span className="text-slate-200 font-bold">{item.dwell}</span>
            </div>
          </div>
        ))}
      </div>

      {/* State Transition Matrix Heatmap */}
      <div className="rounded-xl bg-slatepanel-900 border border-slate-800 p-6 space-y-6">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 border-b border-slate-800 pb-4">
          <div>
            <h2 className="text-base font-semibold text-white flex items-center space-x-2">
              <BarChart3 className="w-5 h-5 text-cyan-400" />
              <span>Empirical State Transition Probability Matrix (P_ij)</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Probabilities P(S_t+1 = j | S_t = i) evaluated across 29,436 sliding windows over the official FD004 test fleet.
            </p>
          </div>
          <div className="text-xs font-mono text-slate-400 px-3 py-1 rounded bg-charcoal-900 border border-slate-800">
            NMI with Regimes: <span className="text-emerald-400 font-bold">{data?.signature_regime_nmi?.toFixed(6) || '0.000143'}</span>
          </div>
        </div>

        {/* Heatmap Grid */}
        <div className="overflow-x-auto">
          <table className="w-full text-center text-xs font-mono border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2.5 px-4 text-left font-sans">Current State (S_t)</th>
                <th className="py-2.5 px-4 text-emerald-400">Next State 0</th>
                <th className="py-2.5 px-4 text-cyan-400">Next State 1</th>
                <th className="py-2.5 px-4 text-amber-400">Next State 2</th>
                <th className="py-2.5 px-4 text-red-400">Next State 3</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {data?.transition_matrix?.map((row, i) => (
                <tr key={i} className="hover:bg-slate-800/40 transition-colors">
                  <td className="py-3 px-4 text-left font-sans font-semibold text-slate-300">
                    State {i} ({stateDetails[i]?.title.split(':')[1]?.trim() || `Cluster ${i}`})
                  </td>
                  {row.map((val, j) => {
                    const pct = (val * 100).toFixed(1);
                    const intensity = val > 0.9 ? 'bg-red-500/20 text-white font-bold' : val > 0.02 ? 'bg-slate-800 text-slate-200' : 'text-slate-500';
                    return (
                      <td key={j} className={`py-3 px-4 ${intensity}`}>
                        <div className="text-xs">{val.toFixed(4)}</div>
                        <div className="text-[10px] text-slate-400">{pct}%</div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Hazard Rate & Dwell Reconciliation Card */}
      <div className="rounded-xl bg-slatepanel-900 border border-slate-800 p-6 space-y-4">
        <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
          <ShieldAlert className="w-5 h-5 text-amber-400" />
          <h2 className="text-base font-semibold text-white">State 3 Hazard-Rate & Dwell-Time Reconciliation Audit</h2>
        </div>
        <p className="text-xs text-slate-300 leading-relaxed">
          In Phase 4.4 / 4.5 statistical audits, the 33.12-cycle State 3 Markov estimate was rigorously resolved from empirical episode duration. 
          The 33.12-cycle value represents a <span className="text-amber-300 font-semibold">model-based reciprocal operational-exit hazard rate</span> (1,689 risk-set cycles / 51 operational exits), under the constant-hazard and terminal-censoring assumptions.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
          <div className="p-3.5 rounded-lg bg-charcoal-900 border border-slate-800">
            <div className="text-[11px] font-mono text-slate-400">Reciprocal Operational-Exit Hazard</div>
            <div className="text-lg font-bold font-mono text-amber-400 mt-1">33.12 Cycles</div>
            <div className="text-[11px] text-slate-500 mt-0.5">1,689 cycles / 51 operational exits</div>
          </div>
          <div className="p-3.5 rounded-lg bg-charcoal-900 border border-slate-800">
            <div className="text-[11px] font-mono text-slate-400">Observed Operational-Exit Mean</div>
            <div className="text-lg font-bold font-mono text-slate-200 mt-1">1.98 Cycles</div>
            <div className="text-[11px] text-slate-500 mt-0.5">Episodes transitioning back to State 0-2</div>
          </div>
          <div className="p-3.5 rounded-lg bg-charcoal-900 border border-slate-800">
            <div className="text-[11px] font-mono text-slate-400">Terminal Failure Episode Mean</div>
            <div className="text-lg font-bold font-mono text-red-400 mt-1">52.23 Cycles</div>
            <div className="text-[11px] text-slate-500 mt-0.5">Observed terminal episodes leading to engine halt</div>
          </div>
        </div>
      </div>

      {/* Scientific Methodology Note */}
      <div className="rounded-xl bg-slatepanel-900/40 border border-slate-800 p-5 space-y-2 text-xs text-slate-400">
        <div className="font-semibold text-slate-300 flex items-center space-x-1.5">
          <Info className="w-4 h-4 text-slate-400" />
          <span>Scientific Qualifications & Ground Truth Attribution</span>
        </div>
        <p className="leading-relaxed">
          {data?.methodological_note ||
            'Discovered signatures are derived from unsupervised clustering of FailureDNA latent embeddings. While low measured Normalized Mutual Information (NMI = 0.000143) confirms that signature clusters do not track instantaneous operating conditions, this statistical independence does not prove physical component isolation (e.g. distinguishing HPC degradation from Fan degradation) in the absence of verified ground-truth component sub-system labels.'}
        </p>
      </div>
    </div>
  );
};
