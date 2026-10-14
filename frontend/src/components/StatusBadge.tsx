import React from 'react';

interface SignatureBadgeProps {
  state: number;
  size?: 'sm' | 'md' | 'lg';
}

export const SignatureBadge: React.FC<SignatureBadgeProps> = ({ state, size = 'md' }) => {
  const styles: Record<number, { bg: string; text: string; border: string; label: string }> = {
    0: {
      bg: 'bg-emerald-950/60',
      text: 'text-emerald-400',
      border: 'border-emerald-500/30',
      label: 'State 0 (Healthy Baseline)',
    },
    1: {
      bg: 'bg-cyan-950/60',
      text: 'text-cyan-400',
      border: 'border-cyan-500/30',
      label: 'State 1 (Early Degradation)',
    },
    2: {
      bg: 'bg-amber-950/60',
      text: 'text-amber-400',
      border: 'border-amber-500/30',
      label: 'State 2 (Accelerated Wear)',
    },
    3: {
      bg: 'bg-red-950/60',
      text: 'text-red-400',
      border: 'border-red-500/30',
      label: 'State 3 (Terminal Critical)',
    },
  };

  const current = styles[state] || {
    bg: 'bg-slate-800',
    text: 'text-slate-300',
    border: 'border-slate-700',
    label: `State ${state}`,
  };

  const sizeClasses = {
    sm: 'text-[10px] px-2 py-0.5',
    md: 'text-xs px-2.5 py-1',
    lg: 'text-sm px-3.5 py-1.5 font-semibold',
  };

  return (
    <span
      className={`inline-flex items-center font-mono rounded-md border ${current.bg} ${current.text} ${current.border} ${sizeClasses[size]}`}
    >
      <span className="w-1.5 h-1.5 rounded-full bg-current mr-1.5 opacity-80" />
      {current.label}
    </span>
  );
};
