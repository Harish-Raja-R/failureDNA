import React, { useEffect, useState } from 'react';
import { Activity, Cpu, Database, FileText, Layers } from 'lucide-react';
import { fetchHealth } from '../services/api';

interface NavbarProps {
  currentTab: string;
  onSelectTab: (tab: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({ currentTab, onSelectTab }) => {
  const [isError, setIsError] = useState(false);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const data = await fetchHealth();
        setIsError(!data.is_ready);
      } catch {
        setIsError(true);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { id: 'overview', label: 'Overview', icon: Layers },
    { id: 'analysis', label: 'Engine Analysis', icon: Activity },
    { id: 'signatures', label: 'Signature Explorer', icon: Cpu },
    { id: 'benchmark', label: 'Benchmark Lab', icon: Database },
    { id: 'documentation', label: 'Documentation', icon: FileText },
  ];

  return (
    <header className="border-b border-slate-800 bg-charcoal-900/90 backdrop-blur sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Project Title */}
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => onSelectTab('overview')}>
            <div className="w-9 h-9 rounded-lg bg-slatepanel-900 border border-red-500/30 flex items-center justify-center text-red-500 shadow-sm shadow-red-500/10">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-lg text-white tracking-wider font-mono">FAILURE<span className="text-red-500">DNA</span></span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">FD004</span>
              </div>
              <p className="text-[11px] text-slate-400 hidden sm:block">Industrial Time-Series Prognostic Architecture</p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="hidden md:flex space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onSelectTab(item.id)}
                  className={`flex items-center space-x-2 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                    isActive
                      ? 'bg-slate-800 text-white border border-slate-700 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/50'
                  }`}
                >
                  <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-red-400' : 'text-slate-500'}`} />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Live Status Indicator */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2 px-2.5 py-1 rounded-full bg-slatepanel-900 border border-slate-800 text-[11px]">
              <span className="relative flex h-2 w-2">
                <span
                  className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                    isError ? 'bg-red-500' : 'bg-emerald-400'
                  }`}
                ></span>
                <span
                  className={`relative inline-flex rounded-full h-2 w-2 ${
                    isError ? 'bg-red-500' : 'bg-emerald-500'
                  }`}
                ></span>
              </span>
              <span className={isError ? 'text-red-400 font-medium' : 'text-emerald-400 font-medium'}>
                {isError ? 'Degraded' : 'Models Loaded'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile Navigation Row */}
      <div className="md:hidden flex overflow-x-auto px-4 py-2 border-t border-slate-800/80 space-x-2">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSelectTab(item.id)}
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded text-xs whitespace-nowrap ${
                isActive ? 'bg-slate-800 text-white font-medium' : 'text-slate-400'
              }`}
            >
              <Icon className="w-3 h-3" />
              <span>{item.label}</span>
            </button>
          );
        })}
      </div>
    </header>
  );
};
