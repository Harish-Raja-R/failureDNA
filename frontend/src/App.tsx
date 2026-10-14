import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { Overview } from './pages/Overview';
import { EngineAnalysis } from './pages/EngineAnalysis';
import { SignatureExplorer } from './pages/SignatureExplorer';
import { BenchmarkLab } from './pages/BenchmarkLab';
import { Documentation } from './pages/Documentation';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<string>('overview');

  const renderContent = () => {
    switch (currentTab) {
      case 'overview':
        return <Overview onNavigate={setCurrentTab} />;
      case 'analysis':
        return <EngineAnalysis />;
      case 'signatures':
        return <SignatureExplorer />;
      case 'benchmark':
        return <BenchmarkLab />;
      case 'documentation':
        return <Documentation />;
      default:
        return <Overview onNavigate={setCurrentTab} />;
    }
  };

  return (
    <div className="min-h-screen bg-charcoal-950 text-slate-100 flex flex-col font-sans selection:bg-red-500/30 selection:text-white">
      <Navbar currentTab={currentTab} onSelectTab={setCurrentTab} />

      <main className="flex-1 pb-16">{renderContent()}</main>

      <footer className="border-t border-slate-800/80 bg-charcoal-900/60 py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 font-mono">
          <div>
            <span className="font-bold text-slate-400">FailureDNA</span> © 2026. Self-Supervised Discovery of Failure Signatures.
          </div>
          <div className="flex items-center space-x-4 text-[11px]">
            <span>Dataset: NASA C-MAPSS FD004</span>
            <span>•</span>
            <span>Phase 6 Integrated Serving</span>
            <span>•</span>
            <span className="text-emerald-400">All 93 Tests Passing</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
