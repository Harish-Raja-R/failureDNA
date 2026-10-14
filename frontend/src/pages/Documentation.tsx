import React from 'react';
import { BookOpen, Database, Layers, Shield, Terminal } from 'lucide-react';

export const Documentation: React.FC = () => {
  return (
    <div className="space-y-8 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 text-slate-300">
      {/* Header */}
      <div className="border-b border-slate-800 pb-5">
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white font-sans flex items-center space-x-3">
          <BookOpen className="w-7 h-7 text-red-500" />
          <span>FailureDNA System Architecture & Technical Documentation</span>
        </h1>
        <p className="text-slate-400 text-sm mt-1">
          Mathematical formulations, pipeline specifications, C-MAPSS FD004 data dictionary, and REST API integration guide.
        </p>
      </div>

      {/* Architecture Overview */}
      <div className="rounded-xl bg-slatepanel-900 border border-slate-800 p-6 space-y-4">
        <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
          <Layers className="w-5 h-5 text-cyan-400" />
          <h2 className="text-base font-semibold text-white">Full-Stack System Pipeline</h2>
        </div>
        <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
          FailureDNA operates as an integrated, non-leaking end-to-end industrial intelligence stack:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-2">
          <div className="p-4 rounded-lg bg-charcoal-900 border border-slate-800 space-y-2">
            <div className="text-xs font-mono text-cyan-400 font-semibold">1. Regime Normalizer</div>
            <p className="text-xs text-slate-400 leading-relaxed">
              KMeans ($k=6$) detects flight operating regimes from altitude, Mach number, and throttle angle. Standardizes 21 sensor channels within regimes, guarding zero-variance channels.
            </p>
          </div>
          <div className="p-4 rounded-lg bg-charcoal-900 border border-slate-800 space-y-2">
            <div className="text-xs font-mono text-red-400 font-semibold">2. Temporal Transformer</div>
            <p className="text-xs text-slate-400 leading-relaxed">
              4-layer, 8-head self-attention encoder (d_model=128, W=50) extracts permutation-invariant temporal representations using masked mean pooling.
            </p>
          </div>
          <div className="p-4 rounded-lg bg-charcoal-900 border border-slate-800 space-y-2">
            <div className="text-xs font-mono text-amber-400 font-semibold">3. Signature Discovery</div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Unsupervised KMeans clustering ($K=4$) groups latent representations into discrete degradation states, ordered monotonically from healthy operation (State 0) to terminal wear (State 3).
            </p>
          </div>
          <div className="p-4 rounded-lg bg-charcoal-900 border border-slate-800 space-y-2">
            <div className="text-xs font-mono text-emerald-400 font-semibold">4. Calibrated RUL Head</div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Linear Ridge probe ($\alpha=10.0$) maps 128-dimensional representations to remaining useful life cycles, strictly bounded by the standard C-MAPSS piece-wise ceiling ($RUL \le 125$).
            </p>
          </div>
        </div>
      </div>

      {/* Dataset & Sensor Channel Dictionary */}
      <div className="rounded-xl bg-slatepanel-900 border border-slate-800 p-6 space-y-4">
        <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
          <Database className="w-5 h-5 text-amber-400" />
          <h2 className="text-base font-semibold text-white">NASA C-MAPSS FD004 Dataset Specifications</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs text-slate-300">
          <div className="space-y-2">
            <h3 className="font-semibold text-slate-200">Operating Characteristics:</h3>
            <ul className="list-disc list-inside space-y-1 text-slate-400">
              <li><strong>Operating Conditions:</strong> 6 discrete flight regimes (altitude 0–42,000 ft, Mach 0–0.84, TRA 20–100°).</li>
              <li><strong>Fault Modes:</strong> Two degradation mechanisms: High Pressure Compressor (HPC) wear and Fan degradation.</li>
              <li><strong>Training Cohort:</strong> 249 turbofan engines (61,249 cycles) run to terminal functional failure.</li>
              <li><strong>Official Test Cohort:</strong> 248 turbofan engines (41,214 cycles) truncated prior to failure.</li>
            </ul>
          </div>
          <div className="space-y-2">
            <h3 className="font-semibold text-slate-200">Active Sensor Channels (14 Varying Channels):</h3>
            <div className="grid grid-cols-2 gap-1.5 font-mono text-[11px] text-slate-400">
              <div><code>s_2</code>: T24 Total Temp</div>
              <div><code>s_3</code>: T30 Total Temp</div>
              <div><code>s_4</code>: T50 Total Temp</div>
              <div><code>s_7</code>: P30 Total Pressure</div>
              <div><code>s_8</code>: Nf Fan Speed</div>
              <div><code>s_9</code>: Nc Core Speed</div>
              <div><code>s_11</code>: Ps30 Static Pressure</div>
              <div><code>s_12</code>: phi Fuel/Ps30</div>
              <div><code>s_13</code>: NRf Corrected Fan</div>
              <div><code>s_14</code>: NRc Corrected Core</div>
              <div><code>s_15</code>: BPR Bypass Ratio</div>
              <div><code>s_17</code>: htBleed Enthalpy</div>
              <div><code>s_20</code>: W31 HPT Coolant</div>
              <div><code>s_21</code>: W32 LPT Coolant</div>
            </div>
            <p className="text-[11px] text-slate-500 pt-1">
              Note: Sensors <code>s_1, s_5, s_10, s_16, s_19</code> have zero within-regime variance and are constant across operating conditions.
            </p>
          </div>
        </div>
      </div>

      {/* REST API Examples */}
      <div className="rounded-xl bg-slatepanel-900 border border-slate-800 p-6 space-y-4">
        <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
          <Terminal className="w-5 h-5 text-red-400" />
          <h2 className="text-base font-semibold text-white">REST API Usage & Integration</h2>
        </div>
        <p className="text-xs text-slate-400">
          Interactive Swagger UI documentation is available at <code>/docs</code>. Below are production cURL integration examples:
        </p>

        <div className="space-y-4 text-xs font-mono">
          {/* Example 1: Health */}
          <div className="rounded-lg bg-charcoal-900 border border-slate-800 p-4 space-y-2">
            <div className="text-slate-400 font-sans font-semibold">1. Health and Readiness Check</div>
            <pre className="text-slate-300 overflow-x-auto p-2 rounded bg-slate-950/60 border border-slate-800/80">
              curl -X GET "http://localhost:8000/health"
            </pre>
          </div>

          {/* Example 2: Benchmark Results */}
          <div className="rounded-lg bg-charcoal-900 border border-slate-800 p-4 space-y-2">
            <div className="text-slate-400 font-sans font-semibold">2. Retrieve Saved Benchmark Metrics</div>
            <pre className="text-slate-300 overflow-x-auto p-2 rounded bg-slate-950/60 border border-slate-800/80">
              curl -X GET "http://localhost:8000/api/v1/research/benchmark"
            </pre>
          </div>

          {/* Example 3: CSV Upload */}
          <div className="rounded-lg bg-charcoal-900 border border-slate-800 p-4 space-y-2">
            <div className="text-slate-400 font-sans font-semibold">3. Upload Engine Telemetry CSV for Analysis</div>
            <pre className="text-slate-300 overflow-x-auto p-2 rounded bg-slate-950/60 border border-slate-800/80">
              curl -X POST "http://localhost:8000/api/v1/analyze/csv" \
                -F "file=@test_engine.csv"
            </pre>
          </div>
        </div>
      </div>

      {/* Disclaimers & Limitations */}
      <div className="rounded-xl bg-slatepanel-900/40 border border-slate-800 p-6 space-y-3 text-xs text-slate-400">
        <div className="flex items-center space-x-2 text-slate-300 font-semibold">
          <Shield className="w-4 h-4 text-slate-400" />
          <span>Research Prototype Disclaimer & Ethical Considerations</span>
        </div>
        <p className="leading-relaxed">
          FailureDNA is developed as an academic research platform for temporal self-supervised representation learning. 
          Discovered signature states (0, 1, 2, 3) reflect unsupervised latent clusters and must not be treated as confirmed physical 
          component fault diagnoses without physical tear-down inspection or ground-truth subsystem annotations. 
          The official benchmark evaluation verified that FailureDNA does not establish overall statistical superiority over gradient-boosted trees on FD004.
        </p>
      </div>
    </div>
  );
};
