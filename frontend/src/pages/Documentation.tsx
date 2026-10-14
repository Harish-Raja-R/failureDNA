import React, { useState } from 'react';
import {
  BookOpen,
  Copy,
  Check,
} from 'lucide-react';
import { SpotlightCard } from '../components/SpotlightCard';

export const Documentation: React.FC = () => {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  const sections = [
    { id: 'research-question', label: '1. Research Question' },
    { id: 'architecture', label: '2. System Architecture' },
    { id: 'mathematics', label: '3. InfoNCE Formulation' },
    { id: 'dataset-regimes', label: '4. C-MAPSS FD004 & Flight Regimes' },
    { id: 'leakage-rules', label: '5. Zero Data Leakage Rules' },
    { id: 'api-reference', label: '6. REST API Reference' },
    { id: 'limitations', label: '7. Limitations & Future Directions' },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      {/* Editorial Header */}
      <div className="border-b border-[#E7E5E0] pb-6 mb-10">
        <div className="inline-flex items-center space-x-2 text-xs font-mono uppercase tracking-wider text-[#F0448A] font-semibold">
          <BookOpen className="w-3.5 h-3.5" />
          <span>Technical Reference & Specifications</span>
        </div>
        <h1 className="font-display text-3xl sm:text-4xl font-bold tracking-tight text-[#171717] mt-1">
          FailureDNA System Documentation
        </h1>
        <p className="text-sm text-[#737373] max-w-3xl leading-relaxed mt-1">
          Authoritative technical specifications, mathematical foundations, data leakage prevention protocols,
          and REST API integration guides for the FailureDNA industrial AI research platform.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-start">
        {/* Sticky Table of Contents Sidebar */}
        <div className="lg:col-span-3 sticky top-24 hidden lg:block">
          <div className="p-4 rounded-xl bg-white border border-[#E7E5E0] shadow-subtle space-y-3">
            <span className="text-[11px] font-mono uppercase tracking-wider text-[#737373] font-semibold block">
              Contents
            </span>
            <nav className="space-y-1 text-xs">
              {sections.map((sec) => (
                <a
                  key={sec.id}
                  href={`#${sec.id}`}
                  className="block px-2.5 py-1.5 rounded-md text-[#737373] hover:text-[#171717] hover:bg-[#F8F8F5] transition-colors"
                >
                  {sec.label}
                </a>
              ))}
            </nav>
          </div>
        </div>

        {/* Documentation Content Body */}
        <div className="lg:col-span-9 space-y-12">
          {/* Section 1: Research Question */}
          <section id="research-question" className="space-y-4 scroll-mt-24">
            <div className="border-b border-[#E7E5E0] pb-2">
              <span className="text-xs font-mono text-[#F0448A] uppercase font-semibold">// Section 1</span>
              <h2 className="font-display text-2xl font-bold text-[#171717]">Research Problem & Formulation</h2>
            </div>
            <p className="text-sm text-[#737373] leading-relaxed">
              Traditional predictive maintenance paradigms predominantly frame equipment health through supervised
              point regression: <em>&ldquo;What is the remaining useful life (RUL) y &isin; ℝ⁺?&rdquo;</em> This formulation
              fails to address severe label scarcity, catastrophic asset destruction risks, and the multi-stage
              physical reality of wear.
            </p>
            <SpotlightCard className="p-5 bg-white border-[#E7E5E0]">
              <span className="font-mono text-xs uppercase tracking-wider text-[#F0448A] font-semibold block">
                Primary Research Question
              </span>
              <p className="font-display text-base font-semibold text-[#171717] mt-1 leading-snug">
                &ldquo;Can self-supervised temporal representation learning discover distinct, physically meaningful
                machine degradation and failure signatures from multivariate industrial time-series without utilizing
                failure labels, ground-truth RUL, fault-mode annotations, or future trajectory information during
                representation learning?&rdquo;
              </p>
            </SpotlightCard>
          </section>

          {/* Section 2: Architecture */}
          <section id="architecture" className="space-y-4 scroll-mt-24">
            <div className="border-b border-[#E7E5E0] pb-2">
              <span className="text-xs font-mono text-[#F0448A] uppercase font-semibold">// Section 2</span>
              <h2 className="font-display text-2xl font-bold text-[#171717]">Model Architecture Specification</h2>
            </div>
            <p className="text-sm text-[#737373] leading-relaxed">
              FailureDNA uses a deep Pre-LayerNorm Temporal Transformer encoder coupled with a non-linear projection
              head for contrastive representation extraction:
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <SpotlightCard className="p-5 bg-white space-y-2">
                <span className="font-mono text-xs font-semibold text-[#171717]">Encoder Backbone (f_&theta;)</span>
                <ul className="text-xs text-[#737373] space-y-1.5 font-mono">
                  <li>&bull; Architecture: Pre-LN Transformer</li>
                  <li>&bull; Context Window (W): 50 Cycles</li>
                  <li>&bull; Layers (L): 4 Encoder Layers</li>
                  <li>&bull; Attention Heads (H): 8 Heads</li>
                  <li>&bull; Model Dimension (d_model): 128</li>
                  <li>&bull; Feed-Forward Hidden (d_ff): 256</li>
                  <li>&bull; Temporal Pooling: Masked Mean Pooling</li>
                </ul>
              </SpotlightCard>

              <SpotlightCard className="p-5 bg-white space-y-2">
                <span className="font-mono text-xs font-semibold text-[#171717]">Projection Head ($g_\phi$)</span>
                <ul className="text-xs text-[#737373] space-y-1.5 font-mono">
                  <li>&bull; Layer 1: Linear(128 &rarr; 128)</li>
                  <li>&bull; Normalization: LayerNorm(128)</li>
                  <li>&bull; Activation: GELU non-linearity</li>
                  <li>&bull; Layer 2: Linear(128 &rarr; 64)</li>
                  <li>&bull; Output: L2-Hypersphere Normalization</li>
                  <li>&bull; Downstream Estimator: Ridge Probe ($\alpha=10.0$)</li>
                  <li>&bull; Clusterer: Spherical K-Means ($K=4$)</li>
                </ul>
              </SpotlightCard>
            </div>
          </section>

          {/* Section 3: InfoNCE Formulation */}
          <section id="mathematics" className="space-y-4 scroll-mt-24">
            <div className="border-b border-[#E7E5E0] pb-2">
              <span className="text-xs font-mono text-[#F0448A] uppercase font-semibold">// Section 3</span>
              <h2 className="font-display text-2xl font-bold text-[#171717]">
                NT-Xent / InfoNCE Self-Supervised Objective
              </h2>
            </div>
            <p className="text-sm text-[#737373] leading-relaxed">
              Given a temporal window X &isin; R^(W &times; D), two stochastic augmentation operators
              generate correlated views X&#771;_A and X&#771;_B. The contrastive
              loss for positive pair (i, j) in a batch of N windows (2N total views) is formulated as:
            </p>
            <div className="p-4 rounded-xl bg-white border border-[#E7E5E0] font-mono text-xs text-[#171717] overflow-x-auto shadow-subtle">
              <code>
                &ell;_(i, j) = - log [ exp(sim(z_i, z_j) / &tau;) / &sum;_(k=1, k &ne; i)^(2N) exp(sim(z_i, z_k) / &tau;) ]
              </code>
            </div>
            <p className="text-xs text-[#737373] leading-relaxed">
              where sim(u, v) = u&middot;v / (||u|| &middot; ||v||) is cosine similarity and &tau; = 0.07 denotes
              the temperature hyperparameter penalizing hard negative temporal representations.
            </p>
          </section>

          {/* Section 4: C-MAPSS FD004 & Flight Regimes */}
          <section id="dataset-regimes" className="space-y-4 scroll-mt-24">
            <div className="border-b border-[#E7E5E0] pb-2">
              <span className="text-xs font-mono text-[#F0448A] uppercase font-semibold">// Section 4</span>
              <h2 className="font-display text-2xl font-bold text-[#171717]">
                NASA C-MAPSS FD004 & Flight Regimes
              </h2>
            </div>
            <p className="text-sm text-[#737373] leading-relaxed">
              FD004 is the most challenging C-MAPSS benchmark, combining simultaneous High-Pressure Compressor (HPC)
              wear and Fan degradation across 6 distinct flight conditions:
            </p>
            <SpotlightCard className="p-5 bg-white space-y-3">
              <div className="overflow-x-auto">
                <table className="w-full text-xs font-mono text-left">
                  <thead>
                    <tr className="border-b border-[#E7E5E0] text-[#737373] text-[11px]">
                      <th className="py-2 pr-4 font-semibold">Regime ID</th>
                      <th className="py-2 px-3 font-semibold">Altitude (ft)</th>
                      <th className="py-2 px-3 font-semibold">Mach Number</th>
                      <th className="py-2 px-3 font-semibold">Throttle (TRA)</th>
                      <th className="py-2 pl-4 font-semibold">Flight Phase</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#F8F8F5] text-[#171717]">
                    <tr>
                      <td className="py-2 pr-4 font-bold text-[#15966A]">Regime 0</td>
                      <td className="py-2 px-3">0 &ndash; 10,000</td>
                      <td className="py-2 px-3">0.00 &ndash; 0.25</td>
                      <td className="py-2 px-3">100°</td>
                      <td className="py-2 pl-4 text-[#737373]">Sea-Level Takeoff / Climb</td>
                    </tr>
                    <tr>
                      <td className="py-2 pr-4 font-bold text-[#5B46D6]">Regime 1</td>
                      <td className="py-2 px-3">10,000 &ndash; 20,000</td>
                      <td className="py-2 px-3">0.25 &ndash; 0.45</td>
                      <td className="py-2 px-3">100°</td>
                      <td className="py-2 pl-4 text-[#737373]">Low-Altitude Climb</td>
                    </tr>
                    <tr>
                      <td className="py-2 pr-4 font-bold text-[#D97706]">Regime 2</td>
                      <td className="py-2 px-3">20,000 &ndash; 30,000</td>
                      <td className="py-2 px-3">0.50 &ndash; 0.70</td>
                      <td className="py-2 px-3">100°</td>
                      <td className="py-2 pl-4 text-[#737373]">Medium Cruise</td>
                    </tr>
                    <tr>
                      <td className="py-2 pr-4 font-bold text-[#F0448A]">Regime 3</td>
                      <td className="py-2 px-3">35,000 &ndash; 42,000</td>
                      <td className="py-2 px-3">0.70 &ndash; 0.84</td>
                      <td className="py-2 px-3">100°</td>
                      <td className="py-2 pl-4 text-[#737373]">High-Altitude Subsonic Cruise</td>
                    </tr>
                    <tr>
                      <td className="py-2 pr-4 font-bold text-[#737373]">Regime 4</td>
                      <td className="py-2 px-3">25,000 &ndash; 35,000</td>
                      <td className="py-2 px-3">0.60 &ndash; 0.75</td>
                      <td className="py-2 px-3">60°</td>
                      <td className="py-2 pl-4 text-[#737373]">Descent / Throttle Back</td>
                    </tr>
                    <tr>
                      <td className="py-2 pr-4 font-bold text-[#737373]">Regime 5</td>
                      <td className="py-2 px-3">0 &ndash; 15,000</td>
                      <td className="py-2 px-3">0.15 &ndash; 0.35</td>
                      <td className="py-2 px-3">20° &ndash; 40°</td>
                      <td className="py-2 pl-4 text-[#737373]">Low-Throttle Loiter / Approach</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </SpotlightCard>
          </section>

          {/* Section 5: Zero Data Leakage Rules */}
          <section id="leakage-rules" className="space-y-4 scroll-mt-24">
            <div className="border-b border-[#E7E5E0] pb-2">
              <span className="text-xs font-mono text-[#F0448A] uppercase font-semibold">// Section 5</span>
              <h2 className="font-display text-2xl font-bold text-[#171717]">
                Non-Negotiable Data Leakage Prevention Rules
              </h2>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <SpotlightCard className="p-5 bg-white space-y-2">
                <span className="font-mono text-xs font-bold text-[#171717]">1. Engine-Level Partitioning</span>
                <p className="text-xs text-[#737373] leading-relaxed">
                  Data is split strictly at the complete engine trajectory level ($Unit\_ID$), never randomly at the window level.
                </p>
              </SpotlightCard>
              <SpotlightCard className="p-5 bg-white space-y-2">
                <span className="font-mono text-xs font-bold text-[#171717]">2. Zero Cross-Split Windows</span>
                <p className="text-xs text-[#737373] leading-relaxed">
                  No window ever spans across engine trajectory boundaries or across train/validation/test partitions.
                </p>
              </SpotlightCard>
              <SpotlightCard className="p-5 bg-white space-y-2">
                <span className="font-mono text-xs font-bold text-[#171717]">3. Train-Only Scaler Fitting</span>
                <p className="text-xs text-[#737373] leading-relaxed">
                  Normalization parameters, K-Means regime centroids, and variance thresholds are fit exclusively on the training engines.
                </p>
              </SpotlightCard>
              <SpotlightCard className="p-5 bg-white space-y-2">
                <span className="font-mono text-xs font-bold text-[#171717]">4. Zero-Label Pretraining</span>
                <p className="text-xs text-[#737373] leading-relaxed">
                  RUL values, failure cycle counts, and fault modes are strictly prohibited from the self-supervised pretraining loop.
                </p>
              </SpotlightCard>
            </div>
          </section>

          {/* Section 6: REST API Reference */}
          <section id="api-reference" className="space-y-4 scroll-mt-24">
            <div className="border-b border-[#E7E5E0] pb-2">
              <span className="text-xs font-mono text-[#F0448A] uppercase font-semibold">// Section 6</span>
              <h2 className="font-display text-2xl font-bold text-[#171717]">REST API Integration Reference</h2>
            </div>
            <p className="text-sm text-[#737373] leading-relaxed">
              The FailureDNA production service exposes high-performance asynchronous REST endpoints:
            </p>

            <SpotlightCard className="p-5 bg-white space-y-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="font-semibold text-[#171717]">POST /api/v1/analyze</span>
                  <button
                    onClick={() =>
                      copyToClipboard(
                        `curl -X POST http://localhost:8000/api/v1/analyze \\\n  -H "Content-Type: application/json" \\\n  -d '{"unit_id": 1, "sequence": [...]}'`,
                        'curl-analyze'
                      )
                    }
                    className="flex items-center space-x-1 text-[#737373] hover:text-[#171717]"
                  >
                    {copiedCode === 'curl-analyze' ? <Check className="w-3.5 h-3.5 text-[#15966A]" /> : <Copy className="w-3.5 h-3.5" />}
                    <span>{copiedCode === 'curl-analyze' ? 'Copied' : 'Copy cURL'}</span>
                  </button>
                </div>
                <pre className="p-3.5 rounded-lg bg-[#F8F8F5] border border-[#E7E5E0] font-mono text-xs text-[#171717] overflow-x-auto">
{`curl -X POST http://localhost:8000/api/v1/analyze \\
  -H "Content-Type: application/json" \\
  -d '{"unit_id": 1, "sequence": [{"cycle": 1, "setting_1": 42.0, "setting_2": 0.84, ...}]}'`}
                </pre>
              </div>

              <div className="space-y-2 pt-2 border-t border-[#F8F8F5]">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="font-semibold text-[#171717]">POST /api/v1/analyze/csv</span>
                </div>
                <pre className="p-3.5 rounded-lg bg-[#F8F8F5] border border-[#E7E5E0] font-mono text-xs text-[#171717] overflow-x-auto">
{`curl -X POST http://localhost:8000/api/v1/analyze/csv \\
  -F "file=@flight_telemetry.csv"`}
                </pre>
              </div>
            </SpotlightCard>
          </section>

          {/* Section 7: Limitations & Future Directions */}
          <section id="limitations" className="space-y-4 scroll-mt-24">
            <div className="border-b border-[#E7E5E0] pb-2">
              <span className="text-xs font-mono text-[#F0448A] uppercase font-semibold">// Section 7</span>
              <h2 className="font-display text-2xl font-bold text-[#171717]">Limitations & Future Directions</h2>
            </div>
            <p className="text-sm text-[#737373] leading-relaxed">
              While FailureDNA establishes the feasibility of self-supervised failure signature extraction, several
              open challenges remain:
            </p>
            <ul className="text-xs text-[#737373] space-y-2 list-disc pl-5 leading-relaxed">
              <li>
                <strong>Cross-Fleet Transferability:</strong> Representations learned from FD004 are optimized for turbofan
                physics and require domain adaptation before zero-shot transfer to other industrial assets (e.g. gas compressors or wind turbine gearboxes).
              </li>
              <li>
                <strong>Supervised Benchmark Disparity:</strong> XGBoost maintains a statistically significant MAE advantage
                over linear probes on the frozen representations ($\Delta = +2.86$ cycles, $p=0.0150$), demonstrating that
                supervised tree algorithms with domain-engineered aggregates remain highly effective on tabular time-series benchmarks.
              </li>
              <li>
                <strong>Continuous vs Discrete Regimes:</strong> Discovered signatures represent discrete cluster medoids;
                continuous manifold geodesics could provide finer-grained degradation tracking in future research phases.
              </li>
            </ul>
          </section>
        </div>
      </div>
    </div>
  );
};

export default Documentation;
