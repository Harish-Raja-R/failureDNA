import React, { useEffect, useState } from 'react';
import {
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
  Cell,
} from 'recharts';
import { fetchBenchmark } from '../services/api';
import { BenchmarkResponse } from '../types';
import { SpotlightCard } from '../components/SpotlightCard';

export const BenchmarkLab: React.FC = () => {
  const [data, setData] = useState<BenchmarkResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeMetricTab, setActiveMetricTab] = useState<'mae_rmse' | 'nasa' | 'r2'>('mae_rmse');

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

  // Filter models for clear comparative chart presentation
  const chartModels =
    data?.model_comparison
      ?.filter((m) => m.model_key !== 'random_encoder_control' && m.model_key !== 'operating_condition_only')
      .map((m) => ({
        key: m.model_key,
        name: m.model_name.replace(' (Window Stats)', '').replace(' Baseline', '').replace(' (Supervised)', ''),
        mae: Number(m.mae.toFixed(2)),
        rmse: Number(m.rmse.toFixed(2)),
        r2: Number(m.r2.toFixed(3)),
        nasa_score: Math.round(m.nasa_score),
        isFailureDNA: m.model_key === 'failuredna_regime_aware',
      })) || [];

  return (
    <div className="space-y-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      {/* Editorial Header */}
      <div className="border-b border-[#E7E5E0] pb-6 flex flex-col sm:flex-row items-start sm:items-end justify-between gap-4">
        <div className="space-y-1.5">
          <div className="inline-flex items-center space-x-2 text-xs font-mono uppercase tracking-wider text-[#F0448A] font-semibold">
            <Database className="w-3.5 h-3.5" />
            <span>Official Fleet Benchmark & Statistical Audit</span>
          </div>
          <h1 className="font-display text-3xl sm:text-4xl font-bold tracking-tight text-[#171717]">
            NASA C-MAPSS FD004 Benchmark Evaluation
          </h1>
          <p className="text-sm text-[#737373] max-w-3xl leading-relaxed">
            Standardized comparative evaluation across all 248 official test engines using 130-cycle early
            degradation capping, accompanied by paired bootstrap inference ($B=10,000$) and Holm FWER control.
          </p>
        </div>

        <button
          onClick={loadData}
          disabled={loading}
          className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-white hover:bg-[#F8F8F5] text-[#171717] text-xs font-medium border border-[#E7E5E0] shadow-subtle transition-all disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 text-[#737373] ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh Benchmark Data</span>
        </button>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-[#FDE8F1] border border-[#F0448A]/30 text-xs text-[#DC2626]">
          {error}
        </div>
      )}

      {/* Official Scientific Verdict Banner */}
      <section>
        <SpotlightCard className="p-6 bg-white border-[#E7E5E0] space-y-4 shadow-card">
          <div className="flex items-center justify-between border-b border-[#E7E5E0] pb-3">
            <div className="flex items-center space-x-2 text-xs font-mono font-semibold text-[#171717]">
              <Scale className="w-4 h-4 text-[#F0448A]" />
              <span className="uppercase tracking-wider">Formal Hypothesis Testing & Statistical Verdict</span>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#F8F8F5] border border-[#E7E5E0] text-[#737373]">
              B = 10,000 Resamples &bull; Seed 42
            </span>
          </div>

          <p className="text-sm sm:text-base text-[#171717] leading-relaxed font-sans font-medium">
            {data?.scientific_conclusion ||
              'Under the omnibus Holm step-down correction for family-wise error rate (FWER) control, XGBoost maintains a statistically significant MAE advantage over FailureDNA (Regime-Aware) (Δ = +2.8571 cycles, adjusted p = 0.0150). Neither the RMSE difference (Δ = +1.2827 cycles, adjusted p = 0.8140) nor the NASA score difference (Δ = -137.12 points, adjusted p = 0.9220) is statistically significant. Therefore, FailureDNA does not establish overall prognostic superiority over gradient-boosted trees on FD004.'}
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2 text-xs font-mono text-[#737373]">
            <div className="p-3 rounded-lg bg-[#F8F8F5] border border-[#E7E5E0]">
              <span className="text-[10px] uppercase text-[#737373] block">MAE Metric Delta</span>
              <span className="text-sm font-semibold text-[#F0448A] mt-0.5 block">&Delta; = +2.86 Cycles</span>
              <span className="text-[10px] text-[#737373]">p_adj = 0.0150 (Stat. Significant)</span>
            </div>
            <div className="p-3 rounded-lg bg-[#F8F8F5] border border-[#E7E5E0]">
              <span className="text-[10px] uppercase text-[#737373] block">RMSE Metric Delta</span>
              <span className="text-sm font-semibold text-[#171717] mt-0.5 block">&Delta; = +1.28 Cycles</span>
              <span className="text-[10px] text-[#737373]">p_adj = 0.8140 (Non-Significant)</span>
            </div>
            <div className="p-3 rounded-lg bg-[#F8F8F5] border border-[#E7E5E0]">
              <span className="text-[10px] uppercase text-[#737373] block">NASA Asymmetric Score Delta</span>
              <span className="text-sm font-semibold text-[#15966A] mt-0.5 block">&Delta; = -137.12 Points</span>
              <span className="text-[10px] text-[#737373]">p_adj = 0.9220 (Non-Significant)</span>
            </div>
          </div>
        </SpotlightCard>
      </section>

      {/* Interactive Benchmark Comparison Charts */}
      <section className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#E7E5E0] pb-3">
          <div>
            <h2 className="font-display text-xl font-bold text-[#171717]">
              Multi-Model Performance Comparison
            </h2>
            <p className="text-xs text-[#737373] mt-0.5">
              Evaluating representations across error magnitude (MAE), heavy penalties (RMSE), asymmetric scoring, and variance explained ($R^2$).
            </p>
          </div>

          {/* Metric Selector Tabs */}
          <div className="flex p-1 rounded-lg bg-white border border-[#E7E5E0] shadow-subtle text-xs font-mono">
            <button
              onClick={() => setActiveMetricTab('mae_rmse')}
              className={`px-3 py-1.5 rounded-md transition-colors ${
                activeMetricTab === 'mae_rmse'
                  ? 'bg-[#171717] text-white font-medium'
                  : 'text-[#737373] hover:text-[#171717]'
              }`}
            >
              MAE & RMSE (Cycles)
            </button>
            <button
              onClick={() => setActiveMetricTab('nasa')}
              className={`px-3 py-1.5 rounded-md transition-colors ${
                activeMetricTab === 'nasa'
                  ? 'bg-[#171717] text-white font-medium'
                  : 'text-[#737373] hover:text-[#171717]'
              }`}
            >
              NASA Score
            </button>
            <button
              onClick={() => setActiveMetricTab('r2')}
              className={`px-3 py-1.5 rounded-md transition-colors ${
                activeMetricTab === 'r2'
                  ? 'bg-[#171717] text-white font-medium'
                  : 'text-[#737373] hover:text-[#171717]'
              }`}
            >
              R² Coefficient
            </button>
          </div>
        </div>

        {/* Chart Canvas */}
        <SpotlightCard className="p-6 bg-white">
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              {activeMetricTab === 'mae_rmse' ? (
                <BarChart data={chartModels} margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E7E5E0" vertical={false} />
                  <XAxis
                    dataKey="name"
                    stroke="#737373"
                    fontSize={11}
                    tickLine={false}
                    axisLine={{ stroke: '#E7E5E0' }}
                  />
                  <YAxis
                    stroke="#737373"
                    fontSize={11}
                    tickLine={false}
                    axisLine={false}
                    label={{
                      value: 'Cycles (Lower is Better)',
                      angle: -90,
                      position: 'insideLeft',
                      fill: '#737373',
                      fontSize: 11,
                    }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#FFFFFF',
                      borderColor: '#E7E5E0',
                      borderRadius: '8px',
                      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.05)',
                      fontSize: '12px',
                      fontFamily: 'JetBrains Mono, monospace',
                    }}
                  />
                  <Legend
                    wrapperStyle={{ paddingTop: '10px', fontSize: '12px', fontFamily: 'JetBrains Mono' }}
                  />
                  <Bar dataKey="mae" name="MAE (Cycles)" fill="#171717" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="rmse" name="RMSE (Cycles)" fill="#F0448A" radius={[4, 4, 0, 0]} />
                </BarChart>
              ) : activeMetricTab === 'nasa' ? (
                <BarChart data={chartModels} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E7E5E0" vertical={false} />
                  <XAxis
                    dataKey="name"
                    stroke="#737373"
                    fontSize={11}
                    tickLine={false}
                    axisLine={{ stroke: '#E7E5E0' }}
                  />
                  <YAxis
                    stroke="#737373"
                    fontSize={11}
                    tickLine={false}
                    axisLine={false}
                    label={{
                      value: 'Asymmetric Penalty (Lower is Better)',
                      angle: -90,
                      position: 'insideLeft',
                      fill: '#737373',
                      fontSize: 11,
                    }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#FFFFFF',
                      borderColor: '#E7E5E0',
                      borderRadius: '8px',
                      fontSize: '12px',
                      fontFamily: 'JetBrains Mono, monospace',
                    }}
                  />
                  <Bar dataKey="nasa_score" name="NASA Penalty Score" fill="#F0448A" radius={[4, 4, 0, 0]}>
                    {chartModels.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.isFailureDNA ? '#F0448A' : entry.key.includes('xgboost') ? '#171717' : '#A3A3A3'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              ) : (
                <BarChart data={chartModels} margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E7E5E0" vertical={false} />
                  <XAxis
                    dataKey="name"
                    stroke="#737373"
                    fontSize={11}
                    tickLine={false}
                    axisLine={{ stroke: '#E7E5E0' }}
                  />
                  <YAxis
                    stroke="#737373"
                    fontSize={11}
                    tickLine={false}
                    axisLine={false}
                    domain={[0, 1]}
                    label={{
                      value: 'R² Score (Higher is Better)',
                      angle: -90,
                      position: 'insideLeft',
                      fill: '#737373',
                      fontSize: 11,
                    }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#FFFFFF',
                      borderColor: '#E7E5E0',
                      borderRadius: '8px',
                      fontSize: '12px',
                      fontFamily: 'JetBrains Mono, monospace',
                    }}
                  />
                  <Bar dataKey="r2" name="R² Coefficient of Determination" fill="#15966A" radius={[4, 4, 0, 0]}>
                    {chartModels.map((entry, index) => (
                      <Cell
                        key={`cell-r2-${index}`}
                        fill={entry.isFailureDNA ? '#F0448A' : entry.key.includes('xgboost') ? '#171717' : '#737373'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              )}
            </ResponsiveContainer>
          </div>
        </SpotlightCard>
      </section>

      {/* Comprehensive Official Metric Table */}
      <section className="space-y-4">
        <div className="flex items-center justify-between border-b border-[#E7E5E0] pb-3">
          <h2 className="font-display text-xl font-bold text-[#171717]">
            Official Test Fleet Performance Table
          </h2>
          <span className="font-mono text-xs text-[#737373]">248 Turbofan Engines</span>
        </div>

        <SpotlightCard className="bg-white overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-[#E7E5E0] bg-[#F8F8F5] text-[#737373] text-[11px]">
                  <th className="py-3 px-4 font-semibold text-[#171717]">Model Architecture</th>
                  <th className="py-3 px-3 font-normal">Feature Type</th>
                  <th className="py-3 px-3 font-normal">MAE (95% CI)</th>
                  <th className="py-3 px-3 font-normal">RMSE (95% CI)</th>
                  <th className="py-3 px-3 font-normal">R² Score</th>
                  <th className="py-3 px-4 font-normal text-right">NASA Score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#E7E5E0]">
                {data?.model_comparison?.map((row) => {
                  const isFailureDNA = row.model_key === 'failuredna_regime_aware';
                  const isXGBoost = row.model_key === 'xgboost_supervised';
                  return (
                    <tr
                      key={row.model_key}
                      className={`hover:bg-[#F8F8F5]/80 transition-colors ${
                        isFailureDNA ? 'bg-[#FDE8F1]/40 font-medium' : isXGBoost ? 'bg-[#F8F8F5]/50' : ''
                      }`}
                    >
                      <td className="py-3 px-4 text-[#171717]">
                        <div className="flex items-center space-x-2">
                          {isFailureDNA && <span className="w-1.5 h-1.5 rounded-full bg-[#F0448A]" />}
                          <span className={isFailureDNA ? 'font-bold text-[#F0448A]' : 'font-medium'}>
                            {row.model_name}
                          </span>
                        </div>
                      </td>
                      <td className="py-3 px-3 text-[#737373] text-[11px]">{row.features}</td>
                      <td className="py-3 px-3 text-[#171717] tabular-nums">
                        {row.mae.toFixed(2)}
                        <span className="text-[#A3A3A3] text-[10px] ml-1">
                          [{row.mae_ci_lower.toFixed(1)}, {row.mae_ci_upper.toFixed(1)}]
                        </span>
                      </td>
                      <td className="py-3 px-3 text-[#171717] tabular-nums">
                        {row.rmse.toFixed(2)}
                        <span className="text-[#A3A3A3] text-[10px] ml-1">
                          [{row.rmse_ci_lower.toFixed(1)}, {row.rmse_ci_upper.toFixed(1)}]
                        </span>
                      </td>
                      <td className="py-3 px-3 text-[#171717] tabular-nums">{row.r2.toFixed(3)}</td>
                      <td className="py-3 px-4 text-right font-semibold tabular-nums text-[#171717]">
                        {Math.round(row.nasa_score).toLocaleString()}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </SpotlightCard>
      </section>

      {/* Paired Bootstrap Statistical Significance Table */}
      {data?.statistical_audit_paired_bootstrap && data.statistical_audit_paired_bootstrap.length > 0 && (
        <section className="space-y-4">
          <div className="flex items-center justify-between border-b border-[#E7E5E0] pb-3">
            <div>
              <h2 className="font-display text-lg font-bold text-[#171717]">
                Paired Bootstrap Inference & FWER Matrix
              </h2>
              <p className="text-xs text-[#737373] mt-0.5">
                Evaluates null hypothesis H₀: E[Δ] = 0 with omnibus Holm-Bonferroni correction.
              </p>
            </div>
            <span className="font-mono text-xs text-[#15966A] font-semibold">Strict Family-Wise Error Control</span>
          </div>

          <SpotlightCard className="bg-white overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead>
                  <tr className="border-b border-[#E7E5E0] bg-[#F8F8F5] text-[#737373] text-[11px]">
                    <th className="py-2.5 px-4 font-semibold text-[#171717]">Comparison Pair</th>
                    <th className="py-2.5 px-3 font-normal">Metric</th>
                    <th className="py-2.5 px-3 font-normal">Delta Mean (Δ)</th>
                    <th className="py-2.5 px-3 font-normal">95% Bootstrap CI</th>
                    <th className="py-2.5 px-3 font-normal">Raw p-value</th>
                    <th className="py-2.5 px-3 font-normal">Holm Omnibus p</th>
                    <th className="py-2.5 px-4 font-normal text-right">Advantage</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#E7E5E0]">
                  {data.statistical_audit_paired_bootstrap.map((row, idx) => (
                    <tr key={idx} className="hover:bg-[#F8F8F5]/60 transition-colors">
                      <td className="py-2.5 px-4 text-[#171717] font-medium">{row.comparison}</td>
                      <td className="py-2.5 px-3 text-[#737373] uppercase">{row.metric}</td>
                      <td className="py-2.5 px-3 tabular-nums font-semibold text-[#171717]">
                        {row.delta_mean > 0 ? `+${row.delta_mean.toFixed(2)}` : row.delta_mean.toFixed(2)}
                      </td>
                      <td className="py-2.5 px-3 tabular-nums text-[#737373]">
                        [{row.ci_lower.toFixed(2)}, {row.ci_upper.toFixed(2)}]
                      </td>
                      <td className="py-2.5 px-3 tabular-nums text-[#737373]">{row.raw_p_value.toFixed(4)}</td>
                      <td className="py-2.5 px-3 tabular-nums font-medium text-[#171717]">
                        {row.holm_adj_p_omnibus.toFixed(4)}
                      </td>
                      <td className="py-2.5 px-4 text-right">
                        <span
                          className={`inline-block px-2 py-0.5 rounded text-[10px] font-semibold ${
                            row.stat_advantage_winner === 'XGBoost'
                              ? 'bg-[#F8F8F5] text-[#171717] border border-[#E7E5E0]'
                              : 'bg-[#EAE5FF] text-[#5B46D6]'
                          }`}
                        >
                          {row.stat_advantage_winner}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SpotlightCard>
        </section>
      )}
    </div>
  );
};

export default BenchmarkLab;
