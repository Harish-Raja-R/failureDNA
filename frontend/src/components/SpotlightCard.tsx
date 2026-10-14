import React, { useRef, useState, useEffect } from 'react';

interface SpotlightCardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  className?: string;
  lift?: boolean;
}

export const SpotlightCard: React.FC<SpotlightCardProps> = ({
  children,
  className = '',
  lift = true,
  ...props
}) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const [disabled, setDisabled] = useState(false);

  useEffect(() => {
    // Disable on touch devices or if reduced motion is requested
    const isTouch = window.matchMedia('(hover: none) and (pointer: coarse)').matches;
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (isTouch || prefersReducedMotion) {
      setDisabled(true);
    }
  }, []);

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (disabled || !cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    cardRef.current.style.setProperty('--mouse-x', `${x}px`);
    cardRef.current.style.setProperty('--mouse-y', `${y}px`);
  };

  return (
    <div
      ref={cardRef}
      onPointerMove={handlePointerMove}
      className={`spotlight-card ${
        lift ? 'hover:-translate-y-0.5 hover:shadow-card hover:border-[#D8D5CE]' : ''
      } ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};

export default SpotlightCard;
