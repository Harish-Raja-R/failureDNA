import React, { useEffect, useState } from 'react';
import { Activity, ArrowUpRight, Cpu, Database, FileText, Layers, Menu, X } from 'lucide-react';
import { fetchHealth } from '../services/api';
import { HealthPill } from './StatusBadge';

interface NavbarProps {
  currentTab: string;
  onSelectTab: (tab: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({ currentTab, onSelectTab }) => {
  const [isReady, setIsReady] = useState(false);
  const [loading, setLoading] = useState(true);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    let isMounted = true;
    const checkHealth = async () => {
      try {
        const data = await fetchHealth();
        if (isMounted) {
          setIsReady(Boolean(data.is_ready));
          setLoading(false);
        }
      } catch {
        if (isMounted) {
          setIsReady(false);
          setLoading(false);
        }
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 25000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const navItems = [
    { id: 'overview', label: 'Overview', icon: Layers },
    { id: 'analysis', label: 'Engine Analysis', icon: Activity },
    { id: 'signatures', label: 'Signature Explorer', icon: Cpu },
    { id: 'benchmark', label: 'Benchmark Lab', icon: Database },
    { id: 'documentation', label: 'Documentation', icon: FileText },
  ];

  return (
    <header className="sticky top-0 z-50 bg-[#F8F8F5]/90 backdrop-blur-md border-b border-[#E7E5E0]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand Mark & FailureDNA Wordmark */}
          <div
            className="flex items-center space-x-3 cursor-pointer group"
            onClick={() => onSelectTab('overview')}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === 'Enter' && onSelectTab('overview')}
            aria-label="FailureDNA Home"
          >
            <div className="w-8 h-8 rounded-lg bg-[#171717] flex items-center justify-center text-white transition-transform group-hover:scale-105 shadow-sm">
              <span className="font-mono text-xs font-bold tracking-tighter">
                F<span className="text-[#F0448A]">D</span>
              </span>
            </div>
            <div className="flex items-baseline space-x-2">
              <span className="font-display font-bold text-base text-[#171717] tracking-tight">
                Failure<span className="text-[#F0448A]">DNA</span>
              </span>
              <span className="text-[10px] font-mono uppercase tracking-widest text-[#737373] hidden sm:inline-block">
                // NASA FD004
              </span>
            </div>
          </div>

          {/* Desktop Navigation Links */}
          <nav className="hidden md:flex items-center space-x-1" aria-label="Main Navigation">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onSelectTab(item.id)}
                  className={`relative flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${
                    isActive
                      ? 'text-[#171717] bg-white border border-[#E7E5E0] shadow-subtle'
                      : 'text-[#737373] hover:text-[#171717] hover:bg-white/60'
                  }`}
                  aria-current={isActive ? 'page' : undefined}
                >
                  <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-[#F0448A]' : 'text-[#737373]'}`} />
                  <span>{item.label}</span>
                  {isActive && (
                    <span className="absolute bottom-0 left-3 right-3 h-[2px] bg-[#F0448A] rounded-full" />
                  )}
                </button>
              );
            })}
          </nav>

          {/* Live Status Indicator & Primary Action */}
          <div className="hidden sm:flex items-center space-x-3">
            <HealthPill isReady={isReady} loading={loading} />

            <button
              onClick={() => onSelectTab('analysis')}
              className="inline-flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg bg-[#171717] hover:bg-[#262626] text-white text-xs font-medium shadow-subtle transition-all duration-200 hover:-translate-y-0.5 active:translate-y-0"
            >
              <span>Analyze Engine</span>
              <ArrowUpRight className="w-3.5 h-3.5 text-[#F0448A]" />
            </button>
          </div>

          {/* Mobile Menu Button */}
          <div className="flex md:hidden items-center space-x-2">
            <HealthPill isReady={isReady} loading={loading} />
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-lg text-[#737373] hover:text-[#171717] hover:bg-white border border-transparent hover:border-[#E7E5E0] transition-colors"
              aria-label="Toggle Navigation Menu"
              aria-expanded={mobileMenuOpen}
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t border-[#E7E5E0] bg-white px-4 py-3 space-y-1 shadow-card animate-fade-in">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => {
                  onSelectTab(item.id);
                  setMobileMenuOpen(false);
                }}
                className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium transition-colors ${
                  isActive
                    ? 'bg-[#FDE8F1] text-[#F0448A] font-semibold'
                    : 'text-[#171717] hover:bg-[#F8F8F5]'
                }`}
              >
                <div className="flex items-center space-x-2.5">
                  <Icon className={`w-4 h-4 ${isActive ? 'text-[#F0448A]' : 'text-[#737373]'}`} />
                  <span>{item.label}</span>
                </div>
                {isActive && <span className="w-1.5 h-1.5 rounded-full bg-[#F0448A]" />}
              </button>
            );
          })}
        </div>
      )}
    </header>
  );
};

export default Navbar;
