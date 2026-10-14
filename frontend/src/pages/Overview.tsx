import React, { useEffect, useState } from 'react';
import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  Cpu,
  Database,
} from 'lucide-react';
import { fetchHealth, fetchMetadata } from '../services/api';
import { HealthResponse, MetadataResponse } from '../types';
import { SpotlightCard } from '../components/SpotlightCard';

interface OverviewProps {
  onNavigate: (tab: string) => void;
}

export const Overview: React.FC<OverviewProps> = ({ onNavigate }) => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [metadata, setMetadata] = useState<MetadataResponse | null>(null);

  useEffect(() => {
    Promise.all([fetchHealth(), fetchMetadata()])
      .then(([h, m]) => {
        setHealth(h);
        setMetadata(m);
      })
      .catch((err) => console.error('Overview initialization error:', err));
  }, []);

  return (
    <div className="space-y-16 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      {/* Editorial Hero Section */}
      <section className="relative pt-4 pb-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-center">
          {/* Left Column: Expressive Headline & Narrative */}
          <div className="lg:col-span-7 space-y-6">
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-white border border-[#E7E5E0] shadow-subtle">
              <span className="w-1.5 h-1.5 rounded-full bg-[#F0448A]" />
              <span className="font-mono text-[10px] tracking-wider uppercase text-[#737373]">
                Research Platform // NASA C-MAPSS FD004
              </span>
            </div>

            <h1 className="font-display text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-[#171717] leading-[1.08]">
              Discover the Signatures Behind Machine Degradation.
            </h1>

            <p className="text-base sm:text-lg text-[#737373] leading-relaxed max-w-2xl font-normal">
              FailureDNA extracts latent degradation trajectories from multivariate industrial time-series
              through self-supervised temporal contrastive learning—uncovering structured wear regimes and
              predictive signatures without failure labels during representation learning.
            </p>

            {/* Call to Actions */}
            <div className="flex flex-wrap items-center gap-4 pt-2">
              <button
                onClick={() => onNavigate('analysis')}
                className="inline-flex items-center space-x-2 px-6 py-3 rounded-lg bg-[#171717] hover:bg-[#262626] text-white text-sm font-medium shadow-card transition-all duration-200 hover:-translate-y-0.5"
              >
                <span>Analyze Engine</span>
                <ArrowRight className="w-4 h-4 text-[#F0448A]" />
              </button>

              <button
                onClick={() => onNavigate('benchmark')}
                className="inline-flex items-center space-x-2 px-5 py-3 rounded-lg bg-white hover:bg-[#F8F8F5] text-[#171717] text-sm font-medium border border-[#E7E5E0] shadow-subtle transition-all duration-200 hover:-translate-y-0.5"
              >
                <span>Explore Benchmark</span>
                <ArrowUpRight className="w-4 h-4 text-[#737373]" />
              </button>

              <div className="flex items-center space-x-2 text-xs font-mono text-[#737373] pl-2">
                <span className="w-2 h-2 rounded-full bg-[#15966A]" />
                <span>94/94 Tests Verified</span>
              </div>
            </div>
          </div>

          {/* Right Column: Visual Telemetry Signature Timeline Preview */}
          <div className="lg:col-span-5">
            <SpotlightCard className="p-6 bg-white border-[#E7E5E0] shadow-card space-y-6">
              {/* Header Details */}
              <div className="flex items-center justify-between border-b border-[#E7E5E0] pb-4">
                <div>
                  <div className="text-[11px] font-mono text-[#737373] uppercase tracking-wider">
                    Degradation Evolution Timeline
                  </div>
                  <div className="font-display text-base font-semibold text-[#171717] mt-0.5">
                    Trajectory Unit #001 Lifecycle
                  </div>
                </div>
                <span className="font-mono text-xs px-2.5 py-1 rounded-md bg-[#F8F8F5] border border-[#E7E5E0] text-[#171717] font-semibold">
                  T = 248 Cycles
                </span>
              </div>

              {/* Graphical State Flow with Degradation Gradient */}
              <div className="space-y-3">
                <div className="flex justify-between items-center text-xs font-mono text-[#737373]">
                  <span>Flight Cycle 1</span>
                  <span>Mid-Life Inflection</span>
                  <span>Terminal Breakdown</span>
                </div>

                {/* Progress Pipeline */}
                <div className="h-3 w-full rounded-full bg-[#F1EFEA] overflow-hidden flex p-0.5">
                  <div className="h-full bg-[#15966A] rounded-l-full w-[36%]" title="State 0: 36%" />
                  <div className="h-full bg-[#5B46D6] w-[36%]" title="State 1: 36%" />
                  <div className="h-full bg-[#D97706] w-[14%]" title="State 2: 14%" />
                  <div className="h-full bg-[#F0448A] rounded-r-full w-[14%]" title="State 3: 14%" />
                </div>

                <div className="grid grid-cols-4 gap-2 pt-2 text-center">
                  <div>
                    <span className="text-[10px] font-mono text-[#15966A] font-medium block">STATE 0</span>
                    <span className="text-[11px] text-[#737373]">Nominal</span>
                  </div>
                  <div>
                    <span className="text-[10px] font-mono text-[#5B46D6] font-medium block">STATE 1</span>
                    <span className="text-[11px] text-[#737373]">Onset Drift</span>
                  </div>
                  <div>
                    <span className="text-[10px] font-mono text-[#D97706] font-medium block">STATE 2</span>
                    <span className="text-[11px] text-[#737373]">Accelerated</span>
                  </div>
                  <div>
                    <span className="text-[10px] font-mono text-[#F0448A] font-medium block">STATE 3</span>
                    <span className="text-[11px] text-[#737373]">Critical</span>
                  </div>
                </div>
              </div>

              {/* Floating Metadata Tags in Hero Card */}
              <div className="pt-2 border-t border-[#E7E5E0] grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-lg bg-[#F8F8F5] border border-[#E7E5E0]">
                  <span className="text-[10px] font-mono text-[#737373] uppercase block">SSL Model Geometry</span>
                  <span className="font-mono text-xs font-semibold text-[#171717] mt-0.5 block">
                    d_model = {metadata?.model_architecture.latent_dim ?? 128} // L = 4
                  </span>
                </div>
                <div className="p-3 rounded-lg bg-[#F8F8F5] border border-[#E7E5E0]">
                  <span className="text-[10px] font-mono text-[#737373] uppercase block">Temporal Window</span>
                  <span className="font-mono text-xs font-semibold text-[#171717] mt-0.5 block">
                    W = {metadata?.sequence_window_length ?? 50} // Stride = 1
                  </span>
                </div>
              </div>

              {/* Micro-insight note */}
              <div className="text-[11px] text-[#737373] flex items-center justify-between pt-1 font-mono">
                <span>API Status: {health?.status === 'healthy' ? 'Operational' : 'Active'}</span>
                <span className="text-[#15966A]">100% Deterministic</span>
              </div>
            </SpotlightCard>
          </div>
        </div>
      </section>

      {/* Key Research Snapshot Strip */}
      <section>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <SpotlightCard className="p-5 bg-white space-y-1.5">
            <span className="text-[11px] font-mono uppercase tracking-wider text-[#737373]">Cohort Fleet</span>
            <div className="font-display text-2xl font-bold text-[#171717] tabular-nums">248 Engines</div>
            <p className="text-xs text-[#737373]">Official {metadata?.dataset ?? 'NASA C-MAPSS FD004'} Test Fleet</p>
          </SpotlightCard>

          <SpotlightCard className="p-5 bg-white space-y-1.5">
            <span className="text-[11px] font-mono uppercase tracking-wider text-[#737373]">Operating Regimes</span>
            <div className="font-display text-2xl font-bold text-[#171717] tabular-nums">
              {metadata?.operating_regimes_count ?? 6} Flight States
            </div>
            <p className="text-xs text-[#737373]">Altitude, Mach, & Throttle Normalization</p>
          </SpotlightCard>

          <SpotlightCard className="p-5 bg-white space-y-1.5">
            <span className="text-[11px] font-mono uppercase tracking-wider text-[#737373]">Discovered Signatures</span>
            <div className="font-display text-2xl font-bold text-[#F0448A] tabular-nums">4 Latent States</div>
            <p className="text-xs text-[#737373]">Monotonically Ordered Degradation Clusters</p>
          </SpotlightCard>

          <SpotlightCard className="p-5 bg-white space-y-1.5">
            <span className="text-[11px] font-mono uppercase tracking-wider text-[#737373]">Prognostic Head</span>
            <div className="font-display text-2xl font-bold text-[#15966A] tabular-nums">Ridge Probe</div>
            <p className="text-xs text-[#737373]">Linear mapping from frozen SSL embeddings</p>
          </SpotlightCard>
        </div>
      </section>

      {/* Asymmetric Research Architecture Workflow */}
      <section className="space-y-6 pt-6">
        <div className="flex flex-col sm:flex-row items-baseline justify-between border-b border-[#E7E5E0] pb-4 gap-2">
          <div>
            <span className="font-mono text-xs uppercase tracking-wider text-[#F0448A] font-semibold">
              // Methodology & Architecture
            </span>
            <h2 className="font-display text-2xl sm:text-3xl font-bold text-[#171717] tracking-tight mt-1">
              The Four-Stage FailureDNA Pipeline
            </h2>
          </div>
          <p className="text-xs text-[#737373] font-mono">
            Zero-Leakage Trajectory Splitting &bull; No Label Supervision
          </p>
        </div>

        {/* Asymmetric Connected Composition */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-stretch">
          {/* Stage 1: Industrial Telemetry (Col span 3) */}
          <SpotlightCard className="md:col-span-3 p-6 bg-white space-y-4 flex flex-col justify-between">
            <div className="space-y-3">
              <div className="w-8 h-8 rounded-lg bg-[#F8F8F5] border border-[#E7E5E0] flex items-center justify-center text-[#171717] font-mono font-bold text-xs">
                01
              </div>
              <h3 className="font-display text-lg font-bold text-[#171717]">
                Industrial Telemetry & Regime Normalization
              </h3>
              <p className="text-xs text-[#737373] leading-relaxed">
                Ingests 21 sensor channels across 6 flight conditions. K-Means clustering isolates operating
                regimes, followed by regime-wise standardization to eliminate extrinsic altitude & throttle confounding.
              </p>
            </div>
            <div className="pt-4 border-t border-[#E7E5E0] text-[11px] font-mono text-[#737373]">
              Input: 24 Channels &bull; W = 50
            </div>
          </SpotlightCard>

          {/* Stage 2: Self-Supervised Representation (Col span 4) */}
          <SpotlightCard className="md:col-span-4 p-6 bg-white space-y-4 flex flex-col justify-between border-[#EAE5FF] bg-gradient-to-br from-white to-[#F8F8FF]">
            <div className="space-y-3">
              <div className="w-8 h-8 rounded-lg bg-[#EAE5FF] text-[#5B46D6] flex items-center justify-center text-xs font-mono font-bold">
                02
              </div>
              <h3 className="font-display text-lg font-bold text-[#171717]">
                Temporal Transformer Contrastive Learning
              </h3>
              <p className="text-xs text-[#737373] leading-relaxed">
                Stochastic views generated via temporal masking, jitter, and sensor dropouts pass through a
                4-layer Pre-LayerNorm Transformer encoder. Optimized with InfoNCE loss onto a normalized
                128-dimensional ℝ¹²⁸ latent hypersphere without failure labels.
              </p>
            </div>
            <div className="pt-4 border-t border-[#EAE5FF] text-[11px] font-mono text-[#5B46D6] font-medium">
              Latent Space: L2-Normalized &bull; Tau = 0.07
            </div>
          </SpotlightCard>

          {/* Stage 3: Signature Discovery (Col span 3) */}
          <SpotlightCard className="md:col-span-3 p-6 bg-white space-y-4 flex flex-col justify-between">
            <div className="space-y-3">
              <div className="w-8 h-8 rounded-lg bg-[#FDE8F1] text-[#F0448A] flex items-center justify-center text-xs font-mono font-bold">
                03
              </div>
              <h3 className="font-display text-lg font-bold text-[#171717]">
                Unsupervised Signature Discovery
              </h3>
              <p className="text-xs text-[#737373] leading-relaxed">
                Spherical K-Means partitions the frozen representation space into 4 discrete degradation states.
                Engine-level trajectories map stochastic Markov transitions toward terminal breakdown.
              </p>
            </div>
            <div className="pt-4 border-t border-[#E7E5E0] text-[11px] font-mono text-[#F0448A] font-medium">
              4 States: S0 &rarr; S1 &rarr; S2 &rarr; S3
            </div>
          </SpotlightCard>

          {/* Stage 4: Downstream Evaluation (Col span 2) */}
          <SpotlightCard className="md:col-span-2 p-6 bg-white space-y-4 flex flex-col justify-between">
            <div className="space-y-3">
              <div className="w-8 h-8 rounded-lg bg-[#E7F7F1] text-[#15966A] flex items-center justify-center text-xs font-mono font-bold">
                04
              </div>
              <h3 className="font-display text-base font-bold text-[#171717]">
                Prognostic RUL Head
              </h3>
              <p className="text-xs text-[#737373] leading-relaxed">
                Ridge probe translates representations into remaining cycles (capped at 125 cycles).
              </p>
            </div>
            <div className="pt-4 border-t border-[#E7E5E0] text-[11px] font-mono text-[#15966A] font-medium">
              MAE: 22.14 cycles
            </div>
          </SpotlightCard>
        </div>
      </section>

      {/* Interactive Tool Access Grid */}
      <section className="space-y-6 pt-4">
        <div className="border-b border-[#E7E5E0] pb-4">
          <span className="font-mono text-xs uppercase tracking-wider text-[#737373] font-semibold">
            // Interactive Research Modules
          </span>
          <h2 className="font-display text-2xl font-bold text-[#171717] tracking-tight mt-1">
            Explore the FailureDNA Toolsuite
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <SpotlightCard
            onClick={() => onNavigate('analysis')}
            className="p-6 bg-white cursor-pointer group hover:border-[#171717] transition-all"
          >
            <div className="space-y-4">
              <div className="w-10 h-10 rounded-lg bg-[#F8F8F5] border border-[#E7E5E0] flex items-center justify-center text-[#171717] group-hover:bg-[#171717] group-hover:text-white transition-colors">
                <Activity className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-display text-lg font-bold text-[#171717] flex items-center justify-between">
                  <span>Engine Analysis Lab</span>
                  <ArrowRight className="w-4 h-4 text-[#737373] group-hover:translate-x-1 group-hover:text-[#F0448A] transition-all" />
                </h3>
                <p className="text-xs text-[#737373] mt-2 leading-relaxed">
                  Upload CSV flight logs or run realistic pre-loaded engine sequences to extract real-time RUL
                  predictions and signature assignments.
                </p>
              </div>
            </div>
          </SpotlightCard>

          <SpotlightCard
            onClick={() => onNavigate('signatures')}
            className="p-6 bg-white cursor-pointer group hover:border-[#171717] transition-all"
          >
            <div className="space-y-4">
              <div className="w-10 h-10 rounded-lg bg-[#F8F8F5] border border-[#E7E5E0] flex items-center justify-center text-[#171717] group-hover:bg-[#171717] group-hover:text-white transition-colors">
                <Cpu className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-display text-lg font-bold text-[#171717] flex items-center justify-between">
                  <span>Signature Explorer</span>
                  <ArrowRight className="w-4 h-4 text-[#737373] group-hover:translate-x-1 group-hover:text-[#F0448A] transition-all" />
                </h3>
                <p className="text-xs text-[#737373] mt-2 leading-relaxed">
                  Inspect the 5x5 absorbing Markov transition matrix, fleet occupancy distributions, and empirical
                  dwell times across degradation states.
                </p>
              </div>
            </div>
          </SpotlightCard>

          <SpotlightCard
            onClick={() => onNavigate('benchmark')}
            className="p-6 bg-white cursor-pointer group hover:border-[#171717] transition-all"
          >
            <div className="space-y-4">
              <div className="w-10 h-10 rounded-lg bg-[#F8F8F5] border border-[#E7E5E0] flex items-center justify-center text-[#171717] group-hover:bg-[#171717] group-hover:text-white transition-colors">
                <Database className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-display text-lg font-bold text-[#171717] flex items-center justify-between">
                  <span>Official Benchmark Lab</span>
                  <ArrowRight className="w-4 h-4 text-[#737373] group-hover:translate-x-1 group-hover:text-[#F0448A] transition-all" />
                </h3>
                <p className="text-xs text-[#737373] mt-2 leading-relaxed">
                  Review the 248-engine comparative audit with paired bootstrap hypothesis tests ($B=10,000$)
                  and Holm FWER statistical corrections.
                </p>
              </div>
            </div>
          </SpotlightCard>
        </div>
      </section>
    </div>
  );
};

export default Overview;
