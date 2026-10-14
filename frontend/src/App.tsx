import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { CustomCursor } from './components/CustomCursor';
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
    <div className="min-h-screen bg-[#F8F8F5] text-[#171717] flex flex-col font-sans selection:bg-[#F0448A]/20 selection:text-[#F0448A] bg-tech-grid relative">
      {/* Subtle non-intrusive custom cursor */}
      <CustomCursor />

      {/* Floating editorial navigation bar */}
      <Navbar currentTab={currentTab} onSelectTab={setCurrentTab} />

      {/* Main Content Area */}
      <main className="flex-1 pb-20">{renderContent()}</main>

      {/* Editorial Footer */}
      <footer className="border-t border-[#E7E5E0] bg-white/80 backdrop-blur py-8 text-xs text-[#737373]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 font-mono">
          <div className="flex items-center space-x-2">
            <span className="font-bold text-[#171717]">Failure<span className="text-[#F0448A]">DNA</span></span>
            <span>&bull;</span>
            <span>19MAM56 CAT-II Industrial AI Research Platform</span>
          </div>

          <div className="flex items-center space-x-4 text-[11px] text-[#737373]">
            <span>NASA C-MAPSS FD004</span>
            <span>&bull;</span>
            <span>248 Official Test Engines</span>
            <span>&bull;</span>
            <span className="inline-flex items-center space-x-1.5 text-[#15966A] font-semibold">
              <span className="w-1.5 h-1.5 rounded-full bg-[#15966A]" />
              <span>94 Tests Passing</span>
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
