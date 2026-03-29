"use client";

import React, { useState, useEffect } from 'react';
import WealthProjectionChart from './components/WealthProjectionChart';
import HouseholdAlignment from './components/HouseholdAlignment';
import MentorChat from './components/MentorChat';
import CouplePlannerUI from './components/CouplePlannerUI';
import PortfolioXRayUI from './components/PortfolioXRayUI';
import PremiumPageUI from './components/PremiumPageUI';

export default function Dashboard() {
  const [files, setFiles] = useState<File[]>([]);
  const [analysis, setAnalysis] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isClient, setIsClient] = useState(false);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [coupleData, setCoupleData] = useState<any>(null);
  const [portfolioData, setPortfolioData] = useState<any>(null);

  // 🏁 THE FIX: Integrated your live Railway URL directly as the primary endpoint.
  const API_BASE_URL = "https://etgen-production.up.railway.app";

  useEffect(() => {
    setIsClient(true);
  }, []);

  const handleUpload = async () => {
    if (files.length === 0) return;
    setLoading(true);
    setAnalysis(null);
    setError(null);

    const formData = new FormData();
    // Correctly sends the file using the 'files' key for the backend
    files.forEach(f => formData.append("files", f));

    try {
      const response = await fetch(`${API_BASE_URL}/api/optimize`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (data.error) throw new Error(data.details || data.error);
      setAnalysis(data);
    } catch (err: any) {
      setError(err.message || "An error occurred.");
    } finally {
      setLoading(false);
    }
  };

  const handleTabSwitch = async (tab: string) => {
    setActiveTab(tab);
    if (tab === 'couple' && !coupleData) {
      const res = await fetch(`${API_BASE_URL}/api/couple/analyze`, { method: "POST" });
      setCoupleData(await res.json());
    } else if (tab === 'portfolio' && !portfolioData) {
      const res = await fetch(`${API_BASE_URL}/api/portfolio/analyze`, { method: "POST" });
      setPortfolioData(await res.json());
    }
  };

  if (!isClient) return null;

  return (
    <div className="min-h-screen bg-[#F0F2F5] text-gray-900 font-sans p-4 md:p-8">
      <div className="max-w-5xl mx-auto space-y-6">
        
        {/* Header Bar */}
        <header className="bg-white px-8 py-4 rounded-xl shadow-sm border border-gray-100 flex justify-between items-center">
          <div className="flex items-center gap-3">
             <div className="w-8 h-8 bg-[#ED1C24] rounded-md flex items-center justify-center shadow-md">
                <span className="text-white font-black text-xs italic">n</span>
             </div>
             <div>
                <h1 className="text-xl font-serif font-black tracking-tighter text-gray-900 leading-none">ET Sentinel</h1>
                <p className="text-[8px] font-bold text-gray-400 uppercase tracking-widest mt-1">AI MONEY MENTOR</p>
             </div>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-[10px] font-bold text-gray-300 uppercase tracking-widest mr-4">PHASE 2 LIVE</span>
            <div className="w-8 h-8 rounded-full bg-gray-900 text-white flex items-center justify-center font-bold text-xs">A</div>
          </div>
        </header>

        {/* Navigation Bar */}
        <nav className="flex gap-1 bg-white p-1 rounded-xl shadow-sm border border-gray-100">
          {[
            { id: 'dashboard', label: 'Dashboard' },
            { id: 'couple', label: "Couple's Planner" },
            { id: 'portfolio', label: 'Portfolio X-Ray' },
            { id: 'premium', label: 'Premium 👑' }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => handleTabSwitch(tab.id)}
              className={`flex-1 py-3 px-4 rounded-lg font-bold text-xs uppercase tracking-widest transition-all ${
                activeTab === tab.id 
                ? 'bg-[#ED1C24] text-white shadow-md' 
                : 'text-gray-400 hover:bg-gray-50'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        {activeTab === 'dashboard' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 grid grid-cols-3 divide-x divide-gray-100">
              <div className="p-6">
                <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest leading-none mb-2">NET WORTH</p>
                <h2 className="text-3xl font-serif font-black text-gray-900 tracking-tighter">
                  {analysis?.household_summary?.total_net_worth || "₹47.3L"}
                </h2>
              </div>
              <div className="p-6">
                <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest leading-none mb-2">MONTHLY SAVINGS</p>
                <h2 className="text-3xl font-serif font-black text-gray-900 tracking-tighter">
                  {analysis?.household_summary?.monthly_savings || "₹59,686"}
                </h2>
              </div>
              <div className="p-6">
                <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest leading-none mb-2">HEALTH SCORE</p>
                <h2 className="text-3xl font-serif font-black text-gray-900 tracking-tighter">
                  {analysis?.household_summary?.health_score || "5.8/10"}
                </h2>
              </div>
            </div>

            <div className="bg-white p-12 rounded-xl shadow-sm border border-gray-100 text-center relative">
              <h2 className="text-3xl font-serif font-black tracking-tighter text-gray-900 mb-2">Upload Your Form 16</h2>
              <p className="text-gray-400 text-sm italic mb-8">Get instant AI tax optimization strategies for your household.</p>
              
              <div className="max-w-xs mx-auto space-y-4">
                 <div className="bg-gray-50 border border-gray-200 rounded-lg p-2 flex items-center gap-3 text-xs text-gray-500 hover:border-gray-400 transition-all cursor-pointer" onClick={() => document.getElementById('file-upload')?.click()}>
                    <span className="text-lg">📄</span>
                    <span className="flex-1 text-left truncate">{files[0]?.name || "Sample_F16.pdf"}</span>
                 </div>
                 <input type="file" id="file-upload" className="hidden" onChange={(e) => setFiles(Array.from(e.target.files || []))} />
                 
                 <button 
                    onClick={handleUpload}
                    disabled={loading}
                    className="w-full bg-black text-white py-4 rounded-lg font-black italic tracking-tight hover:scale-105 transition-all shadow-xl shadow-gray-400/20 uppercase text-xs"
                 >
                    {loading ? "Sentinel analyzing..." : "Analyze Now"}
                 </button>
              </div>

              {analysis && (
                <div className="mt-12 bg-[#F6FFF6] border border-[#E0F0E0] p-10 rounded-3xl animate-in zoom-in duration-500">
                    <p className="text-[10px] font-bold text-[#10B981] uppercase tracking-[0.2em] mb-3">CALCULATED POTENTIAL SAVINGS</p>
                    <h3 className="text-7xl font-serif font-black text-[#10B981] tracking-tighter">
                      {analysis?.tax_optimization?.potential_savings}
                    </h3>
                    <p className="text-lg font-serif italic text-gray-700 mt-6 tracking-tight">
                      " {analysis?.tax_optimization?.primary_action} "
                    </p>
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 h-auto md:h-[350px]">
               <WealthProjectionChart 
                 chartData={analysis?.tax_optimization?.chart_data} 
                 ultimateGain={analysis?.tax_optimization?.ultimate_gain || "₹1.1Cr"} 
               />
               <HouseholdAlignment score={analysis?.household_summary?.household_alignment || 76} />
            </div>

            <div className="pt-12 flex flex-col items-center">
                <h3 className="font-serif font-bold text-xl text-gray-900 flex items-center gap-2 mb-8">
                    <span className="text-[#ED1C24]">●</span> Consult Your Sentinel Mentor
                </h3>
                <div className="w-full max-w-2xl transform">
                    <MentorChat financialContext={analysis?.raw_context || {}} />
                </div>
            </div>
          </div>
        )}

        {activeTab === 'couple' && (
          <CouplePlannerUI data={coupleData} />
        )}

        {activeTab === 'portfolio' && (
          <PortfolioXRayUI data={portfolioData} />
        )}

        {activeTab === 'premium' && (
          <PremiumPageUI />
        )}

        <footer className="pb-12 text-center text-[10px] font-bold text-gray-400 uppercase tracking-widest leading-relaxed max-w-md mx-auto">
            Calculations based on standard Indian Tax slabs and 12% institutional growth projections.
        </footer>

      </div>
    </div>
  );
}
