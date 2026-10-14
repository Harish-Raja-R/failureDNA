import React from 'react';

interface SignatureBadgeProps {
  state: number;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
}

export const SignatureBadge: React.FC<SignatureBadgeProps> = ({
  state,
  size = 'md',
  showLabel = true,
}) => {
  const styles: Record<number, { bg: string; text: string; border: string; dot: string; label: string; name: string }> = {
    0: {
      bg: 'bg-[#E7F7F1]',
      text: 'text-[#15966A]',
      border: 'border-[#15966A]/25',
      dot: 'bg-[#15966A]',
      name: 'State 0',
      label: 'Healthy Baseline',
    },
    1: {
      bg: 'bg-[#EAE5FF]',
      text: 'text-[#5B46D6]',
      border: 'border-[#5B46D6]/25',
      dot: 'bg-[#5B46D6]',
      name: 'State 1',
      label: 'Early Degradation',
    },
    2: {
      bg: 'bg-[#FEF3C7]',
      text: 'text-[#B45309]',
      border: 'border-[#B45309]/25',
      dot: 'bg-[#B45309]',
      name: 'State 2',
      label: 'Accelerated Wear',
    },
    3: {
      bg: 'bg-[#FDE8F1]',
      text: 'text-[#F0448A]',
      border: 'border-[#F0448A]/30',
      dot: 'bg-[#F0448A]',
      name: 'State 3',
      label: 'Terminal Critical',
    },
  };

  const current = styles[state] || {
    bg: 'bg-white',
    text: 'text-[#737373]',
    border: 'border-[#E7E5E0]',
    dot: 'bg-[#737373]',
    name: `State ${state}`,
    label: 'Unknown State',
  };

  const sizeClasses = {
    sm: 'text-[11px] px-2 py-0.5 gap-1.5',
    md: 'text-xs px-2.5 py-1 gap-2 font-medium',
    lg: 'text-sm px-3.5 py-1.5 gap-2.5 font-semibold',
  };

  return (
    <span
      className={`inline-flex items-center font-mono rounded-full border transition-colors ${current.bg} ${current.text} ${current.border} ${sizeClasses[size]}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${current.dot}`} />
      <span className="font-semibold">{current.name}</span>
      {showLabel && (
        <>
          <span className="opacity-40">•</span>
          <span className="font-sans text-[11px] tracking-normal font-normal opacity-90">{current.label}</span>
        </>
      )}
    </span>
  );
};

export const HealthPill: React.FC<{ isReady: boolean; loading?: boolean }> = ({ isReady, loading = false }) => {
  return (
    <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-white border border-[#E7E5E0] shadow-subtle text-xs">
      <span className="relative flex h-2 w-2">
        {isReady && !loading && (
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#15966A] opacity-60" />
        )}
        <span
          className={`relative inline-flex rounded-full h-2 w-2 ${
            loading
              ? 'bg-[#737373]'
              : isReady
              ? 'bg-[#15966A]'
              : 'bg-[#DC2626]'
          }`}
        />
      </span>
      <span className="font-mono text-[11px] text-[#737373]">
        {loading ? (
          'Connecting...'
        ) : isReady ? (
          <span className="text-[#171717] font-medium">Pipeline: <span className="text-[#15966A]">Online</span></span>
        ) : (
          <span className="text-[#DC2626] font-medium">Pipeline: Offline</span>
        )}
      </span>
    </div>
  );
};
