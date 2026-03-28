"use client";

import React from 'react';

export default function PortfolioXRayUI({ data }: { data: any }) {
  if (!data) return <div className="p-12 text-center text-gray-400 italic">Sentinel scanning portfolio assets...</div>;

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div className="bg-white p-12 rounded-3xl border border-gray-100 shadow-sm text-center">
        <div className="w-20 h-20 bg-[#ED1C24]/10 rounded-full flex items-center justify-center mx-auto mb-6">
            <span className="text-4xl">⚡</span>
        </div>
        <h3 className="text-3xl font-serif font-black tracking-tighter text-gray-900 mb-2">Portfolio X-Ray Result</h3>
        <p className="text-gray-400 text-sm italic mb-8">Institutional level audit of your investment architecture.</p>
        
        <div className="grid grid-cols-2 gap-8 max-w-xl mx-auto mb-10">
            <div className="p-6 bg-gray-50 rounded-2xl border border-gray-100">
                <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-2">AVG EXPENSE RATIO</p>
                <h4 className="text-4xl font-serif font-black text-[#ED1C24] tracking-tighter">{data.expense_ratio}</h4>
            </div>
            <div className="p-6 bg-gray-50 rounded-2xl border border-gray-100">
                <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-2">10YR WEALTH LEAKAGE</p>
                <h4 className="text-4xl font-serif font-black text-[#ED1C24] tracking-tighter">{data.leakage_10yr}</h4>
            </div>
        </div>

        <div className="bg-[#FFF8F8] border border-[#FFE0E0] p-6 rounded-2xl inline-block">
             <p className="text-lg font-serif italic text-gray-900 tracking-tight">
                " {data.recommendation} "
             </p>
        </div>
      </div>
    </div>
  );
}
