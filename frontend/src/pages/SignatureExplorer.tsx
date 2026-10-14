import React, { useEffect, useState } from 'react';
import {
  Activity,
  ChevronDown,
  ChevronUp,
  Cpu,
  HelpCircle,
  Info,
  RefreshCw,
} from 'lucide-react';
import { fetchSignatures } from '../services/api';
import { SignaturesResponse } from '../types';
import { SpotlightCard } from '../components/SpotlightCard';
import { SignatureBadge } from '../components/StatusBadge';

export const SignatureExplorer: React.FC = () => {
  const [data, setData] = useState<SignaturesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedState, setSelectedState] = useState<number>(0);
  const [showMethodology, setShowMethodology] = useState(false);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchSignatures();
      setData(res);
    } catch (err: any) {
      setError(err.message || 'Failed loading signature data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  // State Profile Definitions strictly based on Phase 4-5 empirical analysis
  const stateProfiles: Record<
    number,
    {
      title: string;
      subtitle: string;
      empiricalObservations: string[];
      thermodynamicInterpretation: string;
      riskLevel: string;
      meanDwell: string;
      occupancy: string;
      color: string;
    }
  > = {
    0: {
      title: 'State 0: Baseline Nominal Operation',
      subtitle: 'Low-variance trajectory cluster characterized by balanced turbine pressure ratios',
      empiricalObservations: [
        'Predominates initial flight cycles (Cycles 1 through ~80) across all 37 validation trajectories.',
        'Extremely low transition probability to terminal failure (< 0.001 per cycle).',
        'Normalized sensor values remain within ±0.4 standard deviations of flight regime nominals.',
      ],
      thermodynamicInterpretation:
        'Represents undamaged aero-thermodynamic state before observable high-pressure compressor blade erosion or fan blade clearance drift.',
      riskLevel: 'Nominal Operational',
      meanDwell: '28.4 Cycles',
      occupancy: '36.3%',
      color: 'text-[#15966A] bg-[#E7F7F1] border-[#15966A]/20',
    },
    1: {
      title: 'State 1: Early Degradation Onset',
      subtitle: 'First detectable representation shift in compressor exit temperature and pressure ratios',
      empiricalObservations: [
        'Accounts for 36.3% of total flight observations across validation engines.',
        'Primary transition destination from State 0 (P(S0 → S1) = 0.045).',
        'Noticeable divergence in T30 (HPC outlet temperature) while operating conditions remain invariant.',
      ],
      thermodynamicInterpretation:
        'Corresponds to early blade surface roughness accumulation and subtle thermal barrier coating degradation.',
      riskLevel: 'Moderate Wear',
      meanDwell: '28.6 Cycles',
      occupancy: '36.3%',
      color: 'text-[#5B46D6] bg-[#EAE5FF] border-[#5B46D6]/20',
    },
    2: {
      title: 'State 2: Accelerated Mechanical Wear',
      subtitle: 'Intermediate transient phase characterized by fuel flow surge and core speed compensation',
      empiricalObservations: [
        'Transitional cluster with shorter mean dwell time (12.9 cycles).',
        'Exhibits heightened occupancy in engines undergoing simultaneous HPC and Fan dual-mode degradation.',
        'Fuel-to-static-pressure ratio (phi) shifts upward by +1.2σ to maintain commanded engine thrust.',
      ],
      thermodynamicInterpretation:
        'FADEC controller increases fuel metering to compensate for aerodynamic loss in high-pressure compressor stages.',
      riskLevel: 'Elevated Wear',
      meanDwell: '12.9 Cycles',
      occupancy: '7.8%',
      color: 'text-[#B45309] bg-[#FEF3C7] border-[#B45309]/20',
    },
    3: {
      title: 'State 3: Terminal Degradation Regime',
      subtitle: 'Critical pre-failure state preceding functional turbofan shutdown',
      empiricalObservations: [
        'Observed exclusively in the final 30–50 flight cycles before engine retirement.',
        'Direct transition probability to absorbing failure state P(S3 → Failure) = 0.018 per cycle.',
        'Rapid escalation in low-pressure turbine outlet temperature (T50) and static pressure droop.',
      ],
      thermodynamicInterpretation:
        'Terminal mechanical distress and irreversible aerodynamic instability requiring immediate operational teardown.',
      riskLevel: 'Critical Pre-Failure',
      meanDwell: '15.0 Cycles',
      occupancy: '6.8%',
      color: 'text-[#F0448A] bg-[#FDE8F1] border-[#F0448A]/25',
    },
  };

  // State row labels for the 5x5 absorbing transition matrix
  const matrixLabels = ['S0', 'S1', 'S2', 'S3', 'Fail'];

  return (
    <div className="space-y-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      {/* Editorial Header */}
      <div className="border-b border-[#E7E5E0] pb-6 flex flex-col sm:flex-row items-start sm:items-end justify-between gap-4">
        <div className="space-y-1.5">
          <div className="inline-flex items-center space-x-2 text-xs font-mono uppercase tracking-wider text-[#F0448A] font-semibold">
            <Cpu className="w-3.5 h-3.5" />
            <span>Temporal Dynamics & State Modeling</span>
          </div>
          <h1 className="font-display text-3xl sm:text-4xl font-bold tracking-tight text-[#171717]">
            Unsupervised Failure Signature Explorer
          </h1>
          <p className="text-sm text-[#737373] max-w-3xl leading-relaxed">
            Examine discovered degradation regimes, fleet dwell-time statistics, and the empirical absorbing
            Markov state transition matrix derived from 37 complete NASA C-MAPSS FD004 validation trajectories.
          </p>
        </div>

        <button
          onClick={loadData}
          disabled={loading}
          className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-white hover:bg-[#F8F8F5] text-[#171717] text-xs font-medium border border-[#E7E5E0] shadow-subtle transition-all disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 text-[#737373] ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh Artifacts</span>
        </button>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-[#FDE8F1] border border-[#F0448A]/30 text-xs text-[#DC2626]">
          {error}
        </div>
      )}

      {/* Signature State Selector Strip */}
      <section className="space-y-3">
        <div className="flex justify-between items-center text-xs font-mono text-[#737373]">
          <span className="uppercase tracking-wider font-semibold">Select Discovered Cluster</span>
          <span>Click state card to inspect physical profile</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[0, 1, 2, 3].map((stateIdx) => {
            const profile = stateProfiles[stateIdx];
            const isSelected = selectedState === stateIdx;
            return (
              <SpotlightCard
                key={stateIdx}
                onClick={() => setSelectedState(stateIdx)}
                className={`p-5 cursor-pointer text-left transition-all ${
                  isSelected
                    ? 'border-[#171717] ring-1 ring-[#171717] bg-white shadow-card'
                    : 'border-[#E7E5E0] bg-white hover:border-[#D8D5CE]'
                }`}
              >
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <SignatureBadge state={stateIdx} size="sm" showLabel={false} />
                    <span className="font-mono text-xs font-bold text-[#171717]">
                      {profile.occupancy}
                    </span>
                  </div>

                  <div>
                    <h3 className="font-display text-sm font-bold text-[#171717] line-clamp-1">
                      {profile.title.split(':')[1]}
                    </h3>
                    <p className="text-[11px] text-[#737373] mt-1 line-clamp-2">
                      {profile.subtitle}
                    </p>
                  </div>

                  <div className="pt-2 border-t border-[#F8F8F5] flex items-center justify-between text-[11px] font-mono text-[#737373]">
                    <span>Mean Dwell</span>
                    <span className="font-semibold text-[#171717]">{profile.meanDwell}</span>
                  </div>
                </div>
              </SpotlightCard>
            );
          })}
        </div>
      </section>

      {/* Selected State Detailed Scientific Breakdown */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: State Inspection Panel */}
        <div className="lg:col-span-7 space-y-6">
          <SpotlightCard className="p-6 bg-white space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#E7E5E0] pb-4">
              <div className="space-y-1">
                <div className="flex items-center space-x-2">
                  <SignatureBadge state={selectedState} size="md" />
                  <span className="text-xs font-mono text-[#737373]">
                    Occupancy: {stateProfiles[selectedState].occupancy}
                  </span>
                </div>
                <h2 className="font-display text-xl font-bold text-[#171717]">
                  {stateProfiles[selectedState].title}
                </h2>
              </div>

              <div className="text-right">
                <span className="text-[10px] font-mono uppercase text-[#737373] block">Empirical Risk Class</span>
                <span className="text-xs font-mono font-bold text-[#F0448A]">
                  {stateProfiles[selectedState].riskLevel}
                </span>
              </div>
            </div>

            {/* Empirical Observations vs Physical Interpretation */}
            <div className="space-y-5">
              <div className="space-y-2.5">
                <div className="flex items-center space-x-2 text-xs font-mono uppercase tracking-wider text-[#171717] font-semibold">
                  <Activity className="w-3.5 h-3.5 text-[#15966A]" />
                  <span>Empirical Sensor & Latent Observations (Direct Measurement)</span>
                </div>
                <ul className="space-y-2 text-xs text-[#737373] pl-5 list-disc leading-relaxed">
                  {stateProfiles[selectedState].empiricalObservations.map((obs, idx) => (
                    <li key={idx}>{obs}</li>
                  ))}
                </ul>
              </div>

              <div className="p-4 rounded-xl bg-[#F8F8F5] border border-[#E7E5E0] space-y-1.5">
                <div className="flex items-center space-x-2 text-xs font-mono font-semibold text-[#171717]">
                  <HelpCircle className="w-3.5 h-3.5 text-[#F0448A]" />
                  <span>Thermodynamic Interpretation (Domain Physics Hypothesis)</span>
                </div>
                <p className="text-xs text-[#737373] leading-relaxed">
                  {stateProfiles[selectedState].thermodynamicInterpretation}
                </p>
              </div>
            </div>
          </SpotlightCard>
        </div>

        {/* Right Column: 5x5 Absorbing Markov Transition Matrix */}
        <div className="lg:col-span-5 space-y-6">
          <SpotlightCard className="p-6 bg-white space-y-4">
            <div className="flex items-center justify-between border-b border-[#E7E5E0] pb-3">
              <div>
                <h3 className="font-display text-base font-semibold text-[#171717]">
                  State Transition Matrix
                </h3>
                <p className="text-xs text-[#737373] mt-0.5">
                  Absorbing Markov transition probabilities P(i, j)
                </p>
              </div>
              <span className="font-mono text-[10px] uppercase text-[#737373] px-2 py-0.5 rounded bg-[#F8F8F5] border border-[#E7E5E0]">
                5 States
              </span>
            </div>

            {/* Heatmap Matrix Display */}
            {data?.transition_matrix ? (
              <div className="space-y-3 pt-1">
                <div className="overflow-x-auto">
                  <table className="w-full text-center text-xs font-mono border-collapse">
                    <thead>
                      <tr>
                        <th className="p-1.5 text-left text-[11px] text-[#737373]">From \ To</th>
                        {matrixLabels.map((colLabel, colIdx) => (
                          <th key={colIdx} className="p-1.5 text-[11px] font-semibold text-[#171717]">
                            {colLabel}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {data.transition_matrix.map((row, rowIdx) => (
                        <tr key={rowIdx} className="border-t border-[#F1EFEA]">
                          <td className="p-1.5 text-left text-[11px] font-semibold text-[#171717]">
                            {matrixLabels[rowIdx]}
                          </td>
                          {row.map((prob, colIdx) => {
                            const isSelectedRow = rowIdx === selectedState;
                            const bgStyle =
                              prob === 0
                                ? 'bg-[#F8F8F5] text-[#A3A3A3]'
                                : prob > 0.5
                                ? 'bg-[#FDE8F1] text-[#F0448A] font-bold'
                                : prob > 0.1
                                ? 'bg-[#EAE5FF] text-[#5B46D6] font-semibold'
                                : 'bg-[#E7F7F1] text-[#15966A]';

                            return (
                              <td
                                key={colIdx}
                                className={`p-1.5 text-[11px] rounded transition-all ${bgStyle} ${
                                  isSelectedRow ? 'ring-1 ring-[#171717]/20' : ''
                                }`}
                                title={`P(${matrixLabels[rowIdx]} -> ${matrixLabels[colIdx]}) = ${(
                                  prob * 100
                                ).toFixed(2)}%`}
                              >
                                {prob.toFixed(3)}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="pt-2 text-[11px] font-mono text-[#737373] space-y-1">
                  <div className="flex items-center justify-between">
                    <span>Terminal Absorbing State:</span>
                    <span className="font-semibold text-[#171717]">Fail (Engine Teardown)</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Normalized Mutual Info (NMI):</span>
                    <span className="font-semibold text-[#15966A]">
                      {data.signature_regime_nmi !== undefined
                        ? `${(data.signature_regime_nmi * 100).toFixed(2)}% (Regime-Independent)`
                        : '0.0199'}
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-44 flex items-center justify-center text-xs font-mono text-[#737373]">
                Loading transition matrix artifacts...
              </div>
            )}
          </SpotlightCard>

          {/* Research Disclosure Note with Expandable Accordion */}
          <div className="p-4 rounded-xl bg-white border border-[#E7E5E0] space-y-2">
            <button
              onClick={() => setShowMethodology(!showMethodology)}
              className="w-full flex items-center justify-between text-left text-xs font-mono font-semibold text-[#171717]"
            >
              <div className="flex items-center space-x-2">
                <Info className="w-3.5 h-3.5 text-[#F0448A]" />
                <span>Scientific Clarification on Discovered Signatures</span>
              </div>
              {showMethodology ? (
                <ChevronUp className="w-3.5 h-3.5 text-[#737373]" />
              ) : (
                <ChevronDown className="w-3.5 h-3.5 text-[#737373]" />
              )}
            </button>
            <p className="text-xs text-[#737373] leading-relaxed">
              Cluster assignments represent unsupervised partitions of the latent temporal hypersphere, not
              physically verified component failure categories.
            </p>
            {showMethodology && (
              <div className="pt-2 border-t border-[#E7E5E0] text-xs text-[#737373] space-y-2">
                <p>
                  <strong>Observation vs Interpretation:</strong> Dwell times and transition probabilities reflect
                  empirical fleet dynamics across 37 full-lifespan validation units. Physical interpretations
                  (e.g., HPC wear, burner degradation) are hypothesis candidates derived from sensor correlation,
                  not ground-truth teardown logs.
                </p>
                <p>
                  <strong>Absorbing Markov Property:</strong> Once an engine enters the terminal teardown state
                  (Cycle = Lifespan), transition probability to self is 1.0 (P = 1.000).
                </p>
              </div>
            )}
          </div>
        </div>
      </section>
    </div>
  );
};

export default SignatureExplorer;
