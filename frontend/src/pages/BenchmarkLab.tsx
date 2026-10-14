import React, { useEffect, useState } from 'react';
import {
  BarChart2,
  Database,
  RefreshCw,
  Scale,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { fetchBenchmark } from '../services/api';
import { BenchmarkResponse } from '../types';

export const BenchmarkLab: React.FC = () => {
  const [data, setData] = useState<BenchmarkResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeMetricTab, setActiveMetricTab] = useState<'mae_rmse' | 'nasa'>('mae_rmse');

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchBenchmark();
      setData(res);
    } catch (err: any) {
      setError(err.message || 'Failed loading benchmark data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  // Filter models for chart display (primary baselines + FailureDNA models)
  const chartModels = data?.model_comparison
    ?.filter((m) => m.model_key !== 'random_encoder_control' && m.model_key !== 'operating_condition_only')
    .map((m) => ({
      name: m.model_name.replace(' (Window Stats)', '').replace(' Baseline', ''),
      mae: Number(m.mae.toFixed(2)),
      rmse: Number(m.rmse.toFixed(2)),
      r2: Number(m.r2.toFixed(3)),
      nasa_score: Math.round(m.nasa_score),
    })) || [];

  return (
    <div className="space-y-8 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white font-sans flex items-center space-x-3">
            <Database className="w-7 h-7 text-red-500" />
            <span>Official FD004 Benchmark Evaluation & Statistical Audit</span>
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Reproducible comparative evaluation across 248 official NASA C-MAPSS FD004 test engines with FWER-controlled paired bootstrap inference.
          </p>
        </div>
        <button
          onClick={loadData}
          disabled={loading}
          className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-all"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh Metrics</span>
        </button>
      </div>

      {error && (
        <div className="rounded-lg bg-red-950/60 border border-red-500/40 p-4 text-red-300 text-xs">
          Failed to load benchmark data: {error}
        </div>
      )}

      {/* Primary Conclusion Box */}
      <div className="rounded-xl bg-gradient-to-r from-slatepanel-900 via-charcoal-900 to-slatepanel-900 border border-slate-700 p-6 space-y-3">
        <div className="flex items-center space-x-2 text-amber-400 text-xs font-mono font-semibold">
          <Scale className="w-4 h-4" />
          <span>OFFICIAL SCIENTIFIC VERDICT & HYPOTHESIS TESTING SUMMARY</span>
        </div>
        <p className="text-sm sm:text-base text-slate-200 leading-relaxed font-sans font-medium">
          {data?.scientific_conclusion ||
            'Under the omnibus Holm step-down correction for family-wise error rate (FWER) control, XGBoost maintains a statistically significant MAE advantage over FailureDNA (Regime-Aware) (Δ = +2.8571 cycles, adjusted p = 0.0150). Neither the RMSE difference (Δ = +1.2827 cycles, adjusted p = 0.8140) nor the NASA score difference (Δ = -137.12 points, adjusted p = 0.9220) is statistically significant. Therefore, FailureDNA does not establish overall prognostic superiority over gradient-boosted trees on FD004.'}
        </p>
        <div className="flex flex-wrap gap-4 pt-2 text-xs text-slate-400 font-mono">
          <div className="flex items-center space-x-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <span>Direction: Δ = Metric(FailureDNA) − Metric(XGBoost)</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-2 h-2 rounded-full bg-cyan-400" />
            <span>Loss Metrics: Lower is Better (MAE, RMSE, NASA Score)</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-2 h-2 rounded-full bg-amber-400" />
            <span>Goodness-of-Fit: Higher is Better (R²)</span>
          </div>
        </div>
      </div>

      {/* Benchmark Metric Comparison Charts */}
      <div className="rounded-xl bg-slatepanel-900 border border-slate-800 p-6 space-y-6">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
          <div>
            <h2 className="text-base font-semibold text-white flex items-center space-x-2">
              <BarChart2 className="w-5 h-5 text-red-400" />
              <span>Comparative Performance by Architecture (248 Test Engines)</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Comparison across tabular summary baselines, neural baselines, and FailureDNA self-supervised models.
            </p>
          </div>
          <div className="flex rounded-lg bg-charcoal-900 border border-slate-800 p-1 text-xs">
            <button
              onClick={() => setActiveMetricTab('mae_rmse')}
              className={`px-3 py-1 rounded-md font-medium transition-all ${
                activeMetricTab === 'mae_rmse' ? 'bg-slate-800 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              MAE & RMSE (Cycles)
            </button>
            <button
              onClick={() => setActiveMetricTab('nasa')}
              className={`px-3 py-1 rounded-md font-medium transition-all ${
                activeMetricTab === 'nasa' ? 'bg-slate-800 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              NASA Asymmetric Score
            </button>
          </div>
        </div>

        {/* Recharts Bar Chart */}
        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            {activeMetricTab === 'mae_rmse' ? (
              <BarChart data={chartModels} margin={{ top: 20, right: 30, left: 10, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                <XAxis
                  dataKey="name"
                  stroke="#64748B"
                  interval={0}
                  angle={-30}
                  textAnchor="end"
                  tick={{ fill: '#94A3B8', fontSize: 10, fontFamily: 'monospace' }}
                />
                <YAxis stroke="#64748B" tick={{ fill: '#94A3B8', fontSize: 11, fontFamily: 'monospace' }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#131B2E', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                />
                <Legend wrapperStyle={{ paddingTop: '20px' }} />
                <Bar dataKey="mae" name="Mean Absolute Error (Cycles)" fill="#EF4444" radius={[4, 4, 0, 0]} />
                <Bar dataKey="rmse" name="Root Mean Squared Error (Cycles)" fill="#06B6D4" radius={[4, 4, 0, 0]} />
              </BarChart>
            ) : (
              <BarChart data={chartModels} margin={{ top: 20, right: 30, left: 10, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                <XAxis
                  dataKey="name"
                  stroke="#64748B"
                  interval={0}
                  angle={-30}
                  textAnchor="end"
                  tick={{ fill: '#94A3B8', fontSize: 10, fontFamily: 'monospace' }}
                />
                <YAxis stroke="#64748B" tick={{ fill: '#94A3B8', fontSize: 11, fontFamily: 'monospace' }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#131B2E', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                />
                <Legend wrapperStyle={{ paddingTop: '20px' }} />
                <Bar dataKey="nasa_score" name="NASA Asymmetric Penalty Score" fill="#F59E0B" radius={[4, 4, 0, 0]} />
              </BarChart>
            )}
          </ResponsiveContainer>
        </div>
      </div>

      {/* Official Benchmark Leaderboard Table */}
      <div className="rounded-xl bg-slatepanel-900 border border-slate-800 p-6 space-y-4">
        <h3 className="text-base font-semibold text-white">Full Benchmark Evaluation Table (248 Test Engines)</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2.5 px-3 font-sans">Model Architecture</th>
                <th className="py-2.5 px-3 font-sans">Category</th>
                <th className="py-2.5 px-3">MAE [95% CI]</th>
                <th className="py-2.5 px-3">RMSE [95% CI]</th>
                <th className="py-2.5 px-3">R² Score</th>
                <th className="py-2.5 px-3">NASA Penalty</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80">
              {data?.model_comparison?.map((row, idx) => {
                const isWinner = row.model_key === 'xgboost';
                const isFDNA = row.model_key.startsWith('failuredna_regime_aware');
                return (
                  <tr
                    key={idx}
                    className={`hover:bg-slate-800/40 transition-colors ${
                      isWinner ? 'bg-red-950/20 text-red-200' : isFDNA ? 'bg-cyan-950/20 text-cyan-200' : ''
                    }`}
                  >
                    <td className="py-3 px-3 font-sans font-semibold text-slate-200">{row.model_name}</td>
                    <td className="py-3 px-3 text-slate-400 font-sans">{row.category}</td>
                    <td className="py-3 px-3">
                      <span className="font-bold text-white">{row.mae.toFixed(2)}</span>{' '}
                      <span className="text-[10px] text-slate-500">
                        [{row.mae_ci_lower.toFixed(1)}, {row.mae_ci_upper.toFixed(1)}]
                      </span>
                    </td>
                    <td className="py-3 px-3">
                      <span className="font-bold text-white">{row.rmse.toFixed(2)}</span>{' '}
                      <span className="text-[10px] text-slate-500">
                        [{row.rmse_ci_lower.toFixed(1)}, {row.rmse_ci_upper.toFixed(1)}]
                      </span>
                    </td>
                    <td className="py-3 px-3 text-slate-300 font-bold">{row.r2.toFixed(3)}</td>
                    <td className="py-3 px-3 text-amber-300 font-bold">
                      {row.nasa_score > 100000 ? row.nasa_score.toExponential(2) : Math.round(row.nasa_score).toLocaleString()}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Paired Bootstrap Statistical Audit Table */}
      <div className="rounded-xl bg-slatepanel-900 border border-slate-800 p-6 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div>
            <h3 className="text-base font-semibold text-white">
              Phase 5.2.1 Paired-Bootstrap Inference & FWER Multiple-Comparisons Adjustments
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Resampled at engine unit level ($B=2,000$, seed=42). Holm step-down procedure applied across omnibus ($m=15$) and metric ($m=5$) families.
            </p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2.5 px-3 font-sans">Model Comparison</th>
                <th className="py-2.5 px-3">Metric</th>
                <th className="py-2.5 px-3">Delta Mean [95% CI]</th>
                <th className="py-2.5 px-3">Raw p-value</th>
                <th className="py-2.5 px-3">Holm Metric p</th>
                <th className="py-2.5 px-3">Holm Omnibus p</th>
                <th className="py-2.5 px-3 font-sans">Statistical Verdict (α=0.05)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80">
              {data?.statistical_audit_paired_bootstrap?.map((row, idx) => {
                const isSig = row.sig_omnibus_05;
                return (
                  <tr key={idx} className={`hover:bg-slate-800/40 transition-colors ${isSig ? 'bg-red-950/20' : ''}`}>
                    <td className="py-2.5 px-3 font-sans font-medium text-slate-200">{row.comparison}</td>
                    <td className="py-2.5 px-3 uppercase text-slate-400">{row.metric}</td>
                    <td className="py-2.5 px-3">
                      {row.delta_mean >= 0 ? `+${row.delta_mean.toFixed(2)}` : row.delta_mean.toFixed(2)}{' '}
                      <span className="text-[10px] text-slate-500">
                        [{row.ci_lower.toFixed(1)}, {row.ci_upper.toFixed(1)}]
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">{row.raw_p_value.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-slate-300">{row.holm_adj_p_metric.toFixed(4)}</td>
                    <td className="py-2.5 px-3 font-bold text-white">{row.holm_adj_p_omnibus.toFixed(4)}</td>
                    <td className="py-2.5 px-3">
                      {isSig ? (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] bg-red-950 border border-red-500/40 text-red-400 font-bold">
                          Significant ({row.stat_advantage_winner})
                        </span>
                      ) : (
                        <span className="text-[10px] text-slate-500">Not Significant (CI spans 0)</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
