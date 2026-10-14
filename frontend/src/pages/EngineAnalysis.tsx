import React, { useState } from 'react';
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Clock,
  Cpu,
  FileSpreadsheet,
  FileText,
  HelpCircle,
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
import { SpotlightCard } from '../components/SpotlightCard';
import { SignatureBadge } from '../components/StatusBadge';
import { DEMO_TELEMETRY } from '../components/DemoData';

export const EngineAnalysis: React.FC = () => {
  const [activeMode, setActiveMode] = useState<'demo' | 'csv' | 'json'>('demo');
  const [selectedDemo, setSelectedDemo] = useState<'engine_1' | 'engine_2'>('engine_1');
  const [jsonInput, setJsonInput] = useState<string>('');
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [uploadedRecords, setUploadedRecords] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [activeChartSensor, setActiveChartSensor] = useState<string>('s_3');

  const sensorDescriptions: Record<string, { label: string; unit: string; desc: string }> = {
    s_2: { label: 'T24', unit: '°R', desc: 'LPC Outlet Total Temperature' },
    s_3: { label: 'T30', unit: '°R', desc: 'HPC Outlet Total Temperature' },
    s_4: { label: 'T50', unit: '°R', desc: 'LPT Outlet Total Temperature' },
    s_7: { label: 'P30', unit: 'psia', desc: 'HPC Outlet Total Pressure' },
    s_9: { label: 'Nc', unit: 'rpm', desc: 'Physical Core Rotational Speed' },
    s_11: { label: 'Ps30', unit: 'psia', desc: 'Static Pressure at HPC Outlet' },
    s_12: { label: 'phi', unit: 'pps/psia', desc: 'Fuel Flow Ratio (Wf / Ps30)' },
    s_14: { label: 'NRc', unit: 'rpm', desc: 'Corrected Core Rotational Speed' },
    s_15: { label: 'BPR', unit: 'ratio', desc: 'Turbofan Bypass Ratio' },
  };

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
      setError(err.message || 'Analysis failed. Please verify that the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSampleCSV = () => {
    try {
      const records = DEMO_TELEMETRY.engine_1;
      if (!records || records.length === 0) return;
      const headers = Object.keys(records[0]);
      const csvRows = [
        headers.join(','),
        ...records.map((r: any) => headers.map((h) => r[h] ?? '').join(',')),
      ];
      const csvBlob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
      const sampleFile = new File([csvBlob], 'sample_engine_001.csv', { type: 'text/csv' });
      setFile(sampleFile);
      setUploadedRecords(records);
      setError(null);
    } catch (err: any) {
      setError('Could not generate sample file: ' + err.message);
    }
  };

  const handleUploadCSV = async () => {
    if (!file) {
      setError('Please select or drop a valid .csv or .txt telemetry file to upload.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      // Parse sample rows locally for interactive chart preview
      try {
        const text = await file.text();
        const lines = text.trim().split('\n');
        if (lines.length > 0) {
          const firstLine = lines[0].trim();
          const sep = firstLine.includes(',') ? ',' : /\s+/;
          const rawCols = ['unit_id', 'cycle', 'setting_1', 'setting_2', 'setting_3', ...Array.from({ length: 21 }, (_, i) => `s_${i + 1}`)];
          const firstToken = firstLine.split(sep)[0];
          const isNumeric = !isNaN(parseFloat(firstToken));
          const rowsToParse = (isNumeric ? lines : lines.slice(1)).slice(0, 300);
          const parsedRows = rowsToParse.map((line, idx) => {
            const parts = line.trim().split(sep);
            const obj: any = {};
            if (isNumeric) {
              rawCols.forEach((col, cIdx) => {
                obj[col] = parseFloat(parts[cIdx]) || 0;
              });
            } else {
              const headers = firstLine.split(sep);
              headers.forEach((h, cIdx) => {
                obj[h.trim()] = parseFloat(parts[cIdx]) || 0;
              });
            }
            if (!obj.cycle) obj.cycle = idx + 1;
            return obj;
          });
          setUploadedRecords(parsedRows);
        }
      } catch {
        // Fallback silently if preview parsing fails
      }

      const res = await analyzeCSV(file);
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'CSV analysis failed. Please verify file format.');
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
        throw new Error('Sequence payload must be an array of flight cycle observation objects.');
      }
      const res = await analyzeSequence(unitId, sequence);
      setResult(res);
      setUploadedRecords(sequence);
    } catch (err: any) {
      setError(`JSON Payload Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Prepare chart series from demo or uploaded records
  const recordsForChart =
    (activeMode === 'csv' || activeMode === 'json') && uploadedRecords.length > 0
      ? uploadedRecords
      : DEMO_TELEMETRY[selectedDemo] || [];
  const chartData = recordsForChart.map((r: any, idx: number) => ({
    cycle: r.cycle || r.time_cycles || idx + 1,
    value: Number(r[activeChartSensor]) || 0,
  }));

  return (
    <div className="space-y-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      {/* Editorial Page Header */}
      <div className="border-b border-[#E7E5E0] pb-6">
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div className="space-y-1.5">
            <div className="inline-flex items-center space-x-2 text-xs font-mono uppercase tracking-wider text-[#F0448A] font-semibold">
              <Activity className="w-3.5 h-3.5" />
              <span>Interactive Telemetry Laboratory</span>
            </div>
            <h1 className="font-display text-3xl sm:text-4xl font-bold tracking-tight text-[#171717]">
              Engine Degradation & RUL Prognostics
            </h1>
            <p className="text-sm text-[#737373] max-w-3xl leading-relaxed">
              Ingest raw multi-sensor telemetry to perform real-time self-supervised signature extraction,
              operating regime classification, and remaining useful life (RUL) regression on NASA C-MAPSS FD004.
            </p>
          </div>

          <div className="flex items-center space-x-2 text-xs font-mono text-[#737373]">
            <span className="w-2 h-2 rounded-full bg-[#15966A]" />
            <span>Sub-Millisecond Inference Pipeline</span>
          </div>
        </div>
      </div>

      {/* Main Laboratory Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Input Panel & Ingestion Controls */}
        <div className="lg:col-span-5 space-y-6">
          <SpotlightCard className="p-6 bg-white space-y-6">
            {/* Input Selection Tabs */}
            <div className="space-y-3">
              <label className="text-xs font-mono uppercase tracking-wider text-[#737373] font-semibold block">
                Select Telemetry Input Source
              </label>
              <div className="grid grid-cols-3 gap-1.5 p-1 rounded-lg bg-[#F8F8F5] border border-[#E7E5E0]">
                <button
                  onClick={() => setActiveMode('demo')}
                  className={`py-2 px-3 rounded-md text-xs font-medium transition-all ${
                    activeMode === 'demo'
                      ? 'bg-white text-[#171717] shadow-subtle border border-[#E7E5E0]'
                      : 'text-[#737373] hover:text-[#171717]'
                  }`}
                >
                  Demonstration
                </button>
                <button
                  onClick={() => setActiveMode('csv')}
                  className={`py-2 px-3 rounded-md text-xs font-medium transition-all ${
                    activeMode === 'csv'
                      ? 'bg-white text-[#171717] shadow-subtle border border-[#E7E5E0]'
                      : 'text-[#737373] hover:text-[#171717]'
                  }`}
                >
                  Upload CSV
                </button>
                <button
                  onClick={() => setActiveMode('json')}
                  className={`py-2 px-3 rounded-md text-xs font-medium transition-all ${
                    activeMode === 'json'
                      ? 'bg-white text-[#171717] shadow-subtle border border-[#E7E5E0]'
                      : 'text-[#737373] hover:text-[#171717]'
                  }`}
                >
                  Raw JSON
                </button>
              </div>
            </div>

            {/* Mode A: Demonstration Flight Sequences */}
            {activeMode === 'demo' && (
              <div className="space-y-4 pt-1">
                <div className="text-xs text-[#737373] leading-relaxed">
                  Select a validated full-flight telemetry trajectory from the NASA C-MAPSS FD004 test cohort:
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={() => handleRunDemo('engine_1')}
                    disabled={loading}
                    className={`p-3.5 rounded-lg border text-left transition-all ${
                      selectedDemo === 'engine_1'
                        ? 'border-[#171717] bg-[#F8F8F5] ring-1 ring-[#171717]'
                        : 'border-[#E7E5E0] hover:border-[#D8D5CE] bg-white'
                    }`}
                  >
                    <div className="font-mono text-xs font-bold text-[#171717]">Engine #001</div>
                    <div className="text-[11px] text-[#737373] mt-1">100 Flight Cycles</div>
                    <div className="text-[10px] font-mono text-[#15966A] mt-2 font-medium">Standard Flight Log</div>
                  </button>

                  <button
                    onClick={() => handleRunDemo('engine_2')}
                    disabled={loading}
                    className={`p-3.5 rounded-lg border text-left transition-all ${
                      selectedDemo === 'engine_2'
                        ? 'border-[#171717] bg-[#F8F8F5] ring-1 ring-[#171717]'
                        : 'border-[#E7E5E0] hover:border-[#D8D5CE] bg-white'
                    }`}
                  >
                    <div className="font-mono text-xs font-bold text-[#171717]">Engine #002</div>
                    <div className="text-[11px] text-[#737373] mt-1">75 Flight Cycles</div>
                    <div className="text-[10px] font-mono text-[#F0448A] mt-2 font-medium">High Wear Profile</div>
                  </button>
                </div>

                <button
                  onClick={() => handleRunDemo(selectedDemo)}
                  disabled={loading}
                  className="w-full flex items-center justify-center space-x-2 py-3 rounded-lg bg-[#171717] hover:bg-[#262626] text-white text-xs font-medium shadow-card transition-all disabled:opacity-50"
                >
                  <Play className={`w-3.5 h-3.5 text-[#F0448A] ${loading ? 'animate-spin' : ''}`} />
                  <span>{loading ? 'Evaluating Model Ingestion...' : `Execute Analysis on ${selectedDemo === 'engine_1' ? 'Engine #001' : 'Engine #002'}`}</span>
                </button>
              </div>
            )}

            {/* Mode B: CSV / TXT File Ingestion */}
            {activeMode === 'csv' && (
              <div className="space-y-4 pt-1">
                <label
                  onDragOver={(e) => {
                    e.preventDefault();
                    setIsDragging(true);
                  }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setIsDragging(false);
                    if (e.dataTransfer.files?.[0]) {
                      setFile(e.dataTransfer.files[0]);
                      setError(null);
                    }
                  }}
                  className={`border-2 border-dashed rounded-xl p-6 text-center space-y-3 transition-all block cursor-pointer select-none ${
                    isDragging
                      ? 'border-[#F0448A] bg-[#FDE8F1]/50 scale-[1.01]'
                      : file
                      ? 'border-[#15966A]/50 bg-[#F0FDF4]/40 hover:border-[#15966A]'
                      : 'border-[#E7E5E0] hover:border-[#F0448A]/50 bg-[#F8F8F5]/60 hover:bg-[#F8F8F5]'
                  }`}
                >
                  <input
                    type="file"
                    accept=".csv,.txt"
                    className="sr-only"
                    onChange={(e) => {
                      if (e.target.files?.[0]) {
                        setFile(e.target.files[0]);
                        setError(null);
                      }
                    }}
                  />
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center mx-auto transition-colors ${
                    file ? 'bg-[#DCFCE7] text-[#15966A]' : 'bg-white border border-[#E7E5E0] text-[#737373]'
                  }`}>
                    {file ? <FileSpreadsheet className="w-5 h-5 text-[#15966A]" /> : <Upload className="w-5 h-5 text-[#F0448A]" />}
                  </div>
                  <div>
                    <span className="text-xs font-semibold text-[#171717] hover:text-[#F0448A] transition-colors block">
                      {file ? 'File Loaded — Click or drop to replace' : 'Click anywhere to browse or drop file here'}
                    </span>
                    <p className="text-[11px] text-[#737373] mt-1">
                      Supports raw NASA C-MAPSS logs (.txt) or standard comma-separated (.csv)
                    </p>
                  </div>
                  {file && (
                    <div className="inline-flex items-center space-x-2 px-3 py-1.5 rounded-md bg-white border border-[#E7E5E0] font-mono text-xs text-[#171717] shadow-sm">
                      <span className="font-semibold text-[#15966A]">{file.name}</span>
                      <span className="text-[10px] text-[#737373]">({(file.size / 1024).toFixed(1)} KB)</span>
                    </div>
                  )}
                </label>

                {/* Quick Helper Actions */}
                <div className="flex items-center justify-between gap-2 pt-1">
                  <button
                    type="button"
                    onClick={handleLoadSampleCSV}
                    className="text-[11px] font-mono text-[#F0448A] hover:underline flex items-center space-x-1"
                  >
                    <span>⚡ Quick-load sample NASA log</span>
                  </button>
                  <a
                    href="/sample_telemetry.csv"
                    download="sample_telemetry.csv"
                    className="text-[11px] font-mono text-[#737373] hover:text-[#171717] hover:underline"
                  >
                    Download template (.csv)
                  </a>
                </div>

                <button
                  onClick={handleUploadCSV}
                  disabled={loading}
                  className={`w-full flex items-center justify-center space-x-2 py-3 rounded-lg text-xs font-medium shadow-card transition-all ${
                    file
                      ? 'bg-[#171717] hover:bg-[#262626] text-white cursor-pointer'
                      : 'bg-[#F1EFEA] hover:bg-[#E7E5E0] text-[#171717] border border-[#E7E5E0] cursor-pointer'
                  }`}
                >
                  <Activity className={`w-3.5 h-3.5 ${file ? 'text-[#F0448A]' : 'text-[#737373]'} ${loading ? 'animate-spin' : ''}`} />
                  <span>
                    {loading
                      ? 'Processing Telemetry Ingestion...'
                      : file
                      ? `Analyze ${file.name}`
                      : 'Upload or Select a File Above to Analyze'}
                  </span>
                </button>
              </div>
            )}

            {/* Mode C: Raw JSON Sequence */}
            {activeMode === 'json' && (
              <div className="space-y-4 pt-1">
                <div>
                  <textarea
                    rows={6}
                    value={jsonInput}
                    onChange={(e) => setJsonInput(e.target.value)}
                    placeholder='{"unit_id": 1, "sequence": [{"cycle": 1, "setting_1": 42.0, "s_2": 642.5, ...}]}'
                    className="w-full font-mono text-xs p-3 rounded-lg border border-[#E7E5E0] bg-[#F8F8F5] text-[#171717] focus:bg-white focus:border-[#F0448A] focus:outline-none transition-colors"
                  />
                  <p className="text-[11px] text-[#737373] mt-1">
                    Direct JSON payload adhering to the FailureDNA TelemetryInput schema.
                  </p>
                </div>

                <button
                  onClick={handleRunCustomJSON}
                  disabled={loading || !jsonInput.trim()}
                  className="w-full flex items-center justify-center space-x-2 py-3 rounded-lg bg-[#171717] hover:bg-[#262626] text-white text-xs font-medium shadow-card transition-all disabled:opacity-50"
                >
                  <FileText className={`w-3.5 h-3.5 text-[#F0448A] ${loading ? 'animate-spin' : ''}`} />
                  <span>{loading ? 'Validating Payload...' : 'Submit JSON Observation Matrix'}</span>
                </button>
              </div>
            )}

            {/* Error Message Box */}
            {error && (
              <div className="p-4 rounded-lg bg-[#FDE8F1] border border-[#F0448A]/30 text-xs text-[#DC2626] flex items-start space-x-2.5">
                <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-[#F0448A]" />
                <div className="space-y-1">
                  <div className="font-semibold">Analysis Ingestion Error</div>
                  <div>{error}</div>
                </div>
              </div>
            )}
          </SpotlightCard>

          {/* Model Specification & Requirements Card */}
          <SpotlightCard className="p-5 bg-white space-y-3">
            <div className="flex items-center space-x-2 border-b border-[#E7E5E0] pb-2 text-xs font-mono font-semibold text-[#171717]">
              <HelpCircle className="w-3.5 h-3.5 text-[#F0448A]" />
              <span>Input Requirements & Preprocessing Guarantees</span>
            </div>
            <div className="space-y-2 text-xs text-[#737373] leading-relaxed">
              <div className="flex justify-between items-center py-1 border-b border-[#F8F8F5]">
                <span>Receptive Field</span>
                <span className="font-mono text-[#171717] font-medium">W = 50 Cycles</span>
              </div>
              <div className="flex justify-between items-center py-1 border-b border-[#F8F8F5]">
                <span>Active Channels</span>
                <span className="font-mono text-[#171717] font-medium">24 Total (3 Settings + 21 Sensors)</span>
              </div>
              <div className="flex justify-between items-center py-1 border-b border-[#F8F8F5]">
                <span>Short Trajectories</span>
                <span className="font-mono text-[#171717] font-medium">Chronological Edge-Padding</span>
              </div>
              <div className="flex justify-between items-center py-1">
                <span>RUL Bounds</span>
                <span className="font-mono text-[#171717] font-medium">Standard 130-Cycle Ceiling</span>
              </div>
            </div>
          </SpotlightCard>
        </div>

        {/* Right Column: Output Results Display Region */}
        <div className="lg:col-span-7 space-y-6">
          {loading ? (
            /* Loading Skeleton */
            <SpotlightCard className="p-8 bg-white space-y-6 animate-pulse">
              <div className="h-4 bg-[#F1EFEA] rounded w-1/3" />
              <div className="grid grid-cols-2 gap-4">
                <div className="h-24 bg-[#F8F8F5] rounded-xl" />
                <div className="h-24 bg-[#F8F8F5] rounded-xl" />
              </div>
              <div className="h-56 bg-[#F8F8F5] rounded-xl" />
            </SpotlightCard>
          ) : result ? (
            /* Real Model Output Display */
            <div className="space-y-6">
              {/* Primary Prognosis KPI Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {/* RUL Card */}
                <SpotlightCard className="p-6 bg-white space-y-3 border-[#E7E5E0]">
                  <div className="flex items-center justify-between text-xs text-[#737373] font-mono uppercase tracking-wider">
                    <span>Remaining Useful Life (RUL)</span>
                    <Clock className="w-4 h-4 text-[#15966A]" />
                  </div>
                  <div className="flex items-baseline space-x-2">
                    <span className="font-display text-4xl font-extrabold text-[#171717] tabular-nums">
                      {result.predicted_rul.toFixed(1)}
                    </span>
                    <span className="font-mono text-sm text-[#737373]">Cycles</span>
                  </div>
                  <div className="text-xs text-[#737373] font-mono flex items-center justify-between pt-1 border-t border-[#F8F8F5]">
                    <span>Raw Unclipped: {result.predicted_rul_unclipped.toFixed(1)} cycles</span>
                    {result.rul_cap_applied && (
                      <span className="text-[#D97706] font-medium">Capped (130 Max)</span>
                    )}
                  </div>
                </SpotlightCard>

                {/* Signature Assignment Card */}
                <SpotlightCard className="p-6 bg-white space-y-3 border-[#E7E5E0]">
                  <div className="flex items-center justify-between text-xs text-[#737373] font-mono uppercase tracking-wider">
                    <span>Discovered Wear Signature</span>
                    <Cpu className="w-4 h-4 text-[#F0448A]" />
                  </div>
                  <div>
                    <SignatureBadge state={result.signature_state} size="lg" />
                  </div>
                  <div className="text-xs text-[#737373] font-mono flex items-center justify-between pt-1 border-t border-[#F8F8F5]">
                    <span>Centroid Distance</span>
                    <span className="text-[#171717] font-semibold">
                      {result.distance_to_centroid !== null ? result.distance_to_centroid.toFixed(4) : 'N/A'}
                    </span>
                  </div>
                </SpotlightCard>
              </div>

              {/* Warnings / Padding Notification if trajectory < 50 cycles */}
              {result.padding_applied && (
                <div className="p-4 rounded-xl bg-[#FEF3C7] border border-[#D97706]/30 text-xs text-[#92400E] flex items-start space-x-2.5">
                  <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-[#D97706]" />
                  <div>
                    <span className="font-semibold">Short Trajectory Detected:</span> Sequence length (
                    {result.sequence_length} cycles) is shorter than model context window (50 cycles). Chronological
                    edge-replication padded {result.padded_cycles} cycles to guarantee mathematical validity.
                  </div>
                </div>
              )}

              {/* Interactive Telemetry Sensor Trend Chart */}
              <SpotlightCard className="p-6 bg-white space-y-4">
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border-b border-[#E7E5E0] pb-4">
                  <div>
                    <h3 className="font-display text-base font-semibold text-[#171717]">
                      Observed Sensor Telemetry Trajectory
                    </h3>
                    <p className="text-xs text-[#737373] mt-0.5">
                      Visualizing physical channel {activeChartSensor}:{' '}
                      {sensorDescriptions[activeChartSensor]?.desc || activeChartSensor}
                    </p>
                  </div>

                  {/* Sensor Channel Filter Buttons */}
                  <div className="flex flex-wrap gap-1.5 font-mono text-xs">
                    {['s_2', 's_3', 's_4', 's_7', 's_9', 's_12'].map((sensor) => (
                      <button
                        key={sensor}
                        onClick={() => setActiveChartSensor(sensor)}
                        className={`px-2 py-1 rounded text-[11px] font-medium transition-colors ${
                          activeChartSensor === sensor
                            ? 'bg-[#171717] text-white'
                            : 'bg-[#F8F8F5] text-[#737373] hover:text-[#171717] border border-[#E7E5E0]'
                        }`}
                      >
                        {sensorDescriptions[sensor]?.label || sensor}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Recharts Canvas with Clean Editorial Light Theme */}
                <div className="h-64 w-full pt-2">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E7E5E0" vertical={false} />
                      <XAxis
                        dataKey="cycle"
                        stroke="#737373"
                        fontSize={11}
                        tickLine={false}
                        axisLine={{ stroke: '#E7E5E0' }}
                        tickFormatter={(v) => `C${v}`}
                      />
                      <YAxis
                        stroke="#737373"
                        fontSize={11}
                        tickLine={false}
                        axisLine={false}
                        domain={['auto', 'auto']}
                        width={45}
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
                        formatter={(val: any) => [
                          `${Number(val).toFixed(2)} ${sensorDescriptions[activeChartSensor]?.unit || ''}`,
                          sensorDescriptions[activeChartSensor]?.label || activeChartSensor,
                        ]}
                        labelFormatter={(label) => `Flight Cycle: ${label}`}
                      />
                      <Line
                        type="monotone"
                        dataKey="value"
                        stroke="#F0448A"
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4, fill: '#F0448A', stroke: '#FFFFFF', strokeWidth: 2 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </SpotlightCard>

              {/* Sensor Drift Summary Table */}
              {result.sensor_trends && Object.keys(result.sensor_trends).length > 0 && (
                <SpotlightCard className="p-6 bg-white space-y-4">
                  <div className="flex items-center justify-between border-b border-[#E7E5E0] pb-3">
                    <span className="font-display text-sm font-semibold text-[#171717]">
                      Window Sensor Dynamics Summary
                    </span>
                    <span className="font-mono text-xs text-[#737373]">
                      Window Span: C{result.sequence_length - result.window_size + 1} &rarr; C{result.sequence_length}
                    </span>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs font-mono">
                      <thead>
                        <tr className="border-b border-[#E7E5E0] text-[#737373] text-[11px]">
                          <th className="py-2 pr-4 font-normal">Channel</th>
                          <th className="py-2 px-3 font-normal">Start Value</th>
                          <th className="py-2 px-3 font-normal">End Value</th>
                          <th className="py-2 px-3 font-normal">Min / Max</th>
                          <th className="py-2 pl-4 text-right font-normal">Net Delta (&Delta;)</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[#F8F8F5]">
                        {Object.entries(result.sensor_trends).map(([sensorKey, trend]) => {
                          const isPos = trend.delta > 0;
                          return (
                            <tr key={sensorKey} className="hover:bg-[#F8F8F5]/60 transition-colors">
                              <td className="py-2.5 pr-4 text-[#171717] font-semibold">
                                {sensorDescriptions[sensorKey]?.label || sensorKey}
                                <span className="font-sans font-normal text-[11px] text-[#737373] ml-1.5 hidden sm:inline">
                                  ({sensorDescriptions[sensorKey]?.desc || ''})
                                </span>
                              </td>
                              <td className="py-2.5 px-3 text-[#737373] tabular-nums">{trend.start.toFixed(2)}</td>
                              <td className="py-2.5 px-3 text-[#171717] font-medium tabular-nums">{trend.end.toFixed(2)}</td>
                              <td className="py-2.5 px-3 text-[#737373] tabular-nums">
                                {trend.min.toFixed(1)} / {trend.max.toFixed(1)}
                              </td>
                              <td
                                className={`py-2.5 pl-4 text-right font-semibold tabular-nums ${
                                  isPos ? 'text-[#F0448A]' : 'text-[#15966A]'
                                }`}
                              >
                                {isPos ? `+${trend.delta.toFixed(2)}` : trend.delta.toFixed(2)}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </SpotlightCard>
              )}

              {/* Cryptographic Artifact Fingerprint */}
              <div className="p-4 rounded-xl bg-white border border-[#E7E5E0] flex flex-col sm:flex-row items-start sm:items-center justify-between text-xs font-mono text-[#737373] gap-2">
                <div className="flex items-center space-x-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-[#15966A]" />
                  <span>Model: {result.model_metadata.encoder} &bull; Estimator: {result.model_metadata.estimator}</span>
                </div>
                <div className="text-[10px] text-[#A3A3A3]">
                  SHA-256: {result.model_metadata.checkpoint_sha256.slice(0, 16)}...
                </div>
              </div>
            </div>
          ) : (
            /* Empty State Before Analysis */
            <SpotlightCard className="p-12 bg-white text-center space-y-4 border-dashed border-[#E7E5E0]">
              <div className="w-12 h-12 rounded-xl bg-[#F8F8F5] border border-[#E7E5E0] flex items-center justify-center mx-auto text-[#737373]">
                <Activity className="w-6 h-6 text-[#F0448A]" />
              </div>
              <div className="space-y-1">
                <h3 className="font-display text-base font-semibold text-[#171717]">
                  No Telemetry Analyzed Yet
                </h3>
                <p className="text-xs text-[#737373] max-w-sm mx-auto leading-relaxed">
                  Select a demonstration flight profile on the left or upload a C-MAPSS CSV file to evaluate
                  the self-supervised Transformer representation.
                </p>
              </div>
              <button
                onClick={() => handleRunDemo('engine_1')}
                className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-[#171717] hover:bg-[#262626] text-white text-xs font-medium shadow-subtle transition-all"
              >
                <span>Run Demo Engine #001</span>
                <ArrowRight className="w-3.5 h-3.5 text-[#F0448A]" />
              </button>
            </SpotlightCard>
          )}
        </div>
      </div>
    </div>
  );
};

export default EngineAnalysis;
