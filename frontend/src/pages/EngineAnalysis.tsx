import React, { useState } from 'react';
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Cpu,
  FileSpreadsheet,
  FileText,
  Info,
  Play,
  Upload,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { analyzeCSV, analyzeSequence } from '../services/api';
import { AnalysisResponse } from '../types';
import { SignatureBadge } from '../components/StatusBadge';
import { DEMO_TELEMETRY } from '../components/DemoData';

export const EngineAnalysis: React.FC = () => {
  const [activeMode, setActiveMode] = useState<'demo' | 'csv' | 'json'>('demo');
  const [selectedDemo, setSelectedDemo] = useState<'engine_1' | 'engine_2'>('engine_1');
  const [jsonInput, setJsonInput] = useState<string>('');
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [activeChartSensor, setActiveChartSensor] = useState<string>('s_2');

  const handleRunDemo = async (demoKey: 'engine_1' | 'engine_2') => {
    setSelectedDemo(demoKey);
    setLoading(true);
    setError(null);
    try {
      const records = DEMO_TELEMETRY[demoKey];
      const unitId = demoKey === 'engine_1' ? 1 : 2;
      const res = await analyzeSequence(unitId, records as any);
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const handleUploadCSV = async () => {
    if (!file) {
      setError('Please select a .csv or .txt telemetry file to upload.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await analyzeCSV(file);
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'CSV analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const handleRunCustomJSON = async () => {
    setLoading(true);
    setError(null);
    try {
      const parsed = JSON.parse(jsonInput);
      const unitId = parsed.unit_id || 1;
      const sequence = parsed.sequence || parsed;
      if (!Array.isArray(sequence)) {
        throw new Error('Sequence payload must be an array of cycle observations.');
      }
      const res = await analyzeSequence(unitId, sequence);
      setResult(res);
    } catch (err: any) {
      setError(`JSON Input Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Prepare chart series from demo telemetry or sensor trends
  const currentDemoRecords = DEMO_TELEMETRY[selectedDemo] || [];
  const chartData = currentDemoRecords.map((r: any, idx: number) => ({
    cycle: r.cycle || r.time_cycles || idx + 1,
    value: r[activeChartSensor],
  }));

  const sensorDescriptions: Record<string, string> = {
    s_2: 'T24 - LPC Outlet Total Temp (°R)',
    s_3: 'T30 - HPC Outlet Total Temp (°R)',
    s_4: 'T50 - LPT Outlet Total Temp (°R)',
    s_7: 'P30 - HPC Outlet Pressure (psia)',
    s_8: 'Nf - Physical Fan Speed (rpm)',
    s_9: 'Nc - Physical Core Speed (rpm)',
    s_11: 'Ps30 - Static Pressure at HPC (psia)',
    s_12: 'phi - Ratio of fuel flow to Ps30 (pps/psi)',
    s_13: 'NRf - Corrected Fan Speed (rpm)',
    s_14: 'NRc - Corrected Core Speed (rpm)',
    s_15: 'BPR - Bypass Ratio',
    s_17: 'htBleed - Bleed Enthalpy',
    s_20: 'W31 - HPT Coolant Bleed (lbm/s)',
    s_21: 'W32 - LPT Coolant Bleed (lbm/s)',
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Page Header */}
      <div className="border-b border-slate-800 pb-5">
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white font-sans flex items-center space-x-3">
          <Activity className="w-7 h-7 text-red-500" />
          <span>Turbofan Engine Telemetry Analysis</span>
        </h1>
        <p className="text-slate-400 text-sm mt-1">
          Perform non-leaking preprocessing, temporal Transformer encoding, discrete signature discovery, and calibrated RUL estimation.
        </p>
      </div>

      {/* Input Mode Selector */}
      <div className="flex border-b border-slate-800 space-x-6 text-sm font-medium">
        <button
          onClick={() => setActiveMode('demo')}
          className={`pb-3 transition-colors flex items-center space-x-2 ${
            activeMode === 'demo' ? 'border-b-2 border-red-500 text-white' : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Play className="w-4 h-4" />
          <span>Preloaded Test Engines</span>
        </button>
        <button
          onClick={() => setActiveMode('csv')}
          className={`pb-3 transition-colors flex items-center space-x-2 ${
            activeMode === 'csv' ? 'border-b-2 border-red-500 text-white' : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <FileSpreadsheet className="w-4 h-4" />
          <span>Upload CSV File</span>
        </button>
        <button
          onClick={() => setActiveMode('json')}
          className={`pb-3 transition-colors flex items-center space-x-2 ${
            activeMode === 'json' ? 'border-b-2 border-red-500 text-white' : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <FileText className="w-4 h-4" />
          <span>JSON Sequence Input</span>
        </button>
      </div>

      {/* Input Controls Panel */}
      <div className="rounded-xl bg-slatepanel-900/90 border border-slate-800 p-6 space-y-6">
        {activeMode === 'demo' && (
          <div className="space-y-4">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div>
                <h3 className="text-sm font-semibold text-white">Select a Verified NASA FD004 Test Engine Snippet</h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Pre-extracted 50-cycle trailing telemetry from the official benchmark test set.
                </p>
              </div>
              <div className="flex items-center space-x-3">
                <button
                  onClick={() => handleRunDemo('engine_1')}
                  disabled={loading}
                  className={`px-4 py-2 rounded-lg text-xs font-mono font-medium border transition-all ${
                    selectedDemo === 'engine_1' && result?.unit_id === 1
                      ? 'bg-red-600/20 border-red-500 text-red-300'
                      : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
                  }`}
                >
                  Unit 1 (Ground Truth RUL: 22.0)
                </button>
                <button
                  onClick={() => handleRunDemo('engine_2')}
                  disabled={loading}
                  className={`px-4 py-2 rounded-lg text-xs font-mono font-medium border transition-all ${
                    selectedDemo === 'engine_2' && result?.unit_id === 2
                      ? 'bg-cyan-600/20 border-cyan-500 text-cyan-300'
                      : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
                  }`}
                >
                  Unit 2 (Ground Truth RUL: 39.0)
                </button>
              </div>
            </div>
          </div>
        )}

        {activeMode === 'csv' && (
          <div className="space-y-4">
            <div className="border-2 border-dashed border-slate-700 rounded-xl p-8 text-center hover:border-slate-500 transition-colors bg-charcoal-900/40">
              <Upload className="w-10 h-10 text-slate-400 mx-auto mb-3" />
              <div className="text-sm font-medium text-slate-200">
                {file ? file.name : 'Drag and drop or select a telemetry CSV / TXT file'}
              </div>
              <p className="text-xs text-slate-500 mt-1">
                Accepts C-MAPSS FD004 26-column raw files or CSV with headers (setting_1..3, s_1..21). Max size 10 MB.
              </p>
              <input
                type="file"
                accept=".csv,.txt"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                className="hidden"
                id="csv-file-input"
              />
              <div className="mt-4 flex items-center justify-center space-x-3">
                <label
                  htmlFor="csv-file-input"
                  className="cursor-pointer px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700"
                >
                  Choose File
                </label>
                {file && (
                  <button
                    onClick={handleUploadCSV}
                    disabled={loading}
                    className="px-5 py-2 rounded-lg bg-red-600 hover:bg-red-500 text-white text-xs font-medium transition-all"
                  >
                    {loading ? 'Processing...' : 'Upload & Analyze'}
                  </button>
                )}
              </div>
            </div>
          </div>
        )}

        {activeMode === 'json' && (
          <div className="space-y-4">
            <div>
              <label className="block text-xs font-mono text-slate-400 mb-1">
                JSON Sequence Payload (array of cycle objects with setting_1..3 and s_1..21)
              </label>
              <textarea
                value={jsonInput}
                onChange={(e) => setJsonInput(e.target.value)}
                rows={6}
                placeholder='[{"cycle": 1, "setting_1": -0.0007, "setting_2": -0.0004, "setting_3": 100.0, "s_1": 518.67, ..., "s_21": 23.419}]'
                className="w-full rounded-lg bg-charcoal-900 border border-slate-800 p-3 text-xs font-mono text-slate-200 focus:outline-none focus:border-red-500"
              />
            </div>
            <button
              onClick={handleRunCustomJSON}
              disabled={loading || !jsonInput.trim()}
              className="px-5 py-2 rounded-lg bg-red-600 hover:bg-red-500 text-white text-xs font-medium disabled:opacity-50 transition-all"
            >
              {loading ? 'Validating & Analyzing...' : 'Analyze JSON Sequence'}
            </button>
          </div>
        )}

        {/* Error Banner */}
        {error && (
          <div className="rounded-lg bg-red-950/60 border border-red-500/40 p-4 flex items-start space-x-3 text-red-300 text-xs">
            <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold">Analysis Failed:</span> {error}
            </div>
          </div>
        )}
      </div>

      {/* Inference Results View */}
      {result && (
        <div className="space-y-6">
          {/* Top Result Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* RUL Card */}
            <div className="rounded-xl bg-slatepanel-900 border border-slate-800 p-6 space-y-3">
              <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
                <span>ESTIMATED REMAINING USEFUL LIFE</span>
                <Activity className="w-4 h-4 text-red-400" />
              </div>
              <div className="flex items-baseline space-x-2">
                <span className="text-4xl font-extrabold font-mono text-white tracking-tight">
                  {result.predicted_rul.toFixed(2)}
                </span>
                <span className="text-sm font-mono text-slate-400">CYCLES</span>
              </div>
              <div className="text-xs text-slate-400 pt-1 border-t border-slate-800 flex items-center justify-between">
                <span>Raw Unclipped: {result.predicted_rul_unclipped.toFixed(2)}</span>
                <span className="text-[11px] font-mono text-slate-500">Probe: Ridge (α=10.0)</span>
              </div>
            </div>

            {/* Signature Card */}
            <div className="rounded-xl bg-slatepanel-900 border border-slate-800 p-6 space-y-3">
              <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
                <span>DISCOVERED FAILURE SIGNATURE</span>
                <Cpu className="w-4 h-4 text-cyan-400" />
              </div>
              <div>
                <SignatureBadge state={result.signature_state} size="lg" />
              </div>
              <div className="text-xs text-slate-300 pt-1 border-t border-slate-800">
                <p className="line-clamp-2">{result.signature_description}</p>
                {result.distance_to_centroid !== null && (
                  <p className="text-[11px] font-mono text-slate-500 mt-1">
                    Latent Centroid Distance: {result.distance_to_centroid.toFixed(4)}
                  </p>
                )}
              </div>
            </div>

            {/* Sequence & Operating Regimes Card */}
            <div className="rounded-xl bg-slatepanel-900 border border-slate-800 p-6 space-y-3">
              <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
                <span>ENGINE CONTEXT & FLIGHT CONDITIONS</span>
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              </div>
              <div className="text-sm font-semibold text-slate-200">
                Engine Unit #{result.unit_id} ({result.sequence_length} cycles logged)
              </div>
              <div className="flex flex-wrap gap-1.5 pt-1">
                {result.detected_operating_regimes.map((regimeId) => (
                  <span
                    key={regimeId}
                    className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300"
                  >
                    Regime {regimeId}
                  </span>
                ))}
              </div>
              {result.padding_applied && (
                <div className="text-[11px] text-amber-400 font-mono">
                  Edge-padded: +{result.padded_cycles} cycles to cycle 0
                </div>
              )}
            </div>
          </div>

          {/* Warnings List */}
          {result.warnings && result.warnings.length > 0 && (
            <div className="rounded-xl bg-amber-950/40 border border-amber-500/30 p-4 space-y-1 text-xs text-amber-300">
              <div className="font-semibold flex items-center space-x-1.5">
                <Info className="w-4 h-4 text-amber-400" />
                <span>Inference Pipeline Warnings</span>
              </div>
              <ul className="list-disc list-inside space-y-0.5 text-amber-300/90 pl-1 font-mono text-[11px]">
                {result.warnings.map((w, idx) => (
                  <li key={idx}>{w}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Interactive Sensor Trend Chart */}
          <div className="rounded-xl bg-slatepanel-900 border border-slate-800 p-6 space-y-4">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
              <div>
                <h3 className="text-base font-semibold text-white">Chronological Sensor Trajectory Trend</h3>
                <p className="text-xs text-slate-400">
                  Select an active turbofan sensor channel to visualize temporal trajectory progression.
                </p>
              </div>
              <div className="flex items-center space-x-2">
                <label className="text-xs font-mono text-slate-400">Sensor:</label>
                <select
                  value={activeChartSensor}
                  onChange={(e) => setActiveChartSensor(e.target.value)}
                  className="rounded-lg bg-charcoal-900 border border-slate-700 text-xs font-mono text-slate-200 px-3 py-1.5 focus:outline-none focus:border-red-500"
                >
                  {Object.entries(sensorDescriptions).map(([k, label]) => (
                    <option key={k} value={k}>
                      {k.toUpperCase()} - {label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Recharts Trajectory Plot */}
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                  <XAxis
                    dataKey="cycle"
                    stroke="#64748B"
                    tick={{ fill: '#94A3B8', fontSize: 11, fontFamily: 'monospace' }}
                    label={{ value: 'Flight Cycle', position: 'insideBottomRight', offset: -5, fill: '#64748B', fontSize: 11 }}
                  />
                  <YAxis
                    stroke="#64748B"
                    domain={['auto', 'auto']}
                    tick={{ fill: '#94A3B8', fontSize: 11, fontFamily: 'monospace' }}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#131B2E', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                    labelStyle={{ color: '#F8FAFC', fontFamily: 'monospace' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#EF4444"
                    strokeWidth={2}
                    dot={{ r: 2, fill: '#EF4444' }}
                    activeDot={{ r: 5, fill: '#F8FAFC' }}
                    name={sensorDescriptions[activeChartSensor] || activeChartSensor}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Sensor Trends Table */}
            <div className="pt-4 border-t border-slate-800">
              <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3">
                Key Channel Summary Statistics
              </h4>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-7 gap-3">
                {Object.entries(result.sensor_trends).map(([sensorKey, trend]) => (
                  <div
                    key={sensorKey}
                    onClick={() => setActiveChartSensor(sensorKey)}
                    className={`cursor-pointer p-2.5 rounded-lg border transition-all text-left ${
                      activeChartSensor === sensorKey
                        ? 'bg-red-950/40 border-red-500/50 text-white'
                        : 'bg-charcoal-900 border-slate-800 text-slate-400 hover:border-slate-700'
                    }`}
                  >
                    <div className="text-[10px] font-mono font-semibold uppercase">{sensorKey}</div>
                    <div className="text-xs font-mono font-bold text-slate-200 mt-0.5">{trend.mean.toFixed(1)}</div>
                    <div className="text-[10px] font-mono text-slate-500">
                      Δ {trend.delta >= 0 ? `+${trend.delta.toFixed(2)}` : trend.delta.toFixed(2)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
