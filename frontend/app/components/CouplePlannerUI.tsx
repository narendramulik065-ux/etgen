"use client";

import React from 'react';

export default function CouplePlannerUI({ data }: { data: any }) {
  if (!data) return <div className="p-12 text-center text-gray-400 italic">Sentinel syncing synergy data...</div>;

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      {/* Optimization Banner */}
      <div className="bg-[#10B981] p-10 rounded-3xl text-center text-white shadow-xl">
        <p className="text-[10px] font-bold uppercase tracking-[0.2em] mb-3 opacity-80">JOINT TAX OPTIMIZATION POTENTIAL</p>
        <h3 className="text-6xl font-serif font-black tracking-tighter">{data.summary.total_annual_savings}</h3>
        <p className="text-lg font-serif italic mt-4 opacity-90">"Strategic synergy detected between {data.comparison.partner_a.name} and {data.comparison.partner_b.name}."</p>
      </div>

      {/* Comparison Grid */}
      <div className="grid grid-cols-2 gap-6">
        {[data.comparison.partner_a, data.comparison.partner_b].map((partner, i) => (
          <div key={i} className="bg-white p-8 rounded-2xl border border-gray-100 shadow-sm">
            <h4 className="font-serif font-bold text-2xl text-gray-900 mb-6">{partner.name}</h4>
            <div className="space-y-4">
              <div className="flex justify-between border-b border-gray-50 pb-2">
                <span className="text-[10px] font-bold text-gray-400 uppercase">Income</span>
                <span className="font-bold text-gray-900 font-serif">₹{partner.income}</span>
              </div>
              <div className="flex justify-between border-b border-gray-50 pb-2">
                <span className="text-[10px] font-bold text-gray-400 uppercase">Tax Bracket</span>
                <span className="font-bold text-[#ED1C24]">{partner.tax_bracket}</span>
              </div>
              <div className="flex justify-between border-b border-gray-50 pb-2">
                <span className="text-[10px] font-bold text-gray-400 uppercase">HRA Status</span>
                <span className="font-bold text-gray-900">{partner.hra_status}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[10px] font-bold text-gray-400 uppercase">80C Usage</span>
                <span className="font-bold text-gray-900">{partner['80c_usage']}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Checklist */}
      <div className="bg-white p-8 rounded-2xl border border-gray-100 shadow-sm">
        <h4 className="font-serif font-bold text-xl text-gray-900 mb-6 flex items-center gap-2">
            <span className="text-[#ED1C24]">●</span> Actionable Synergy Checklist
        </h4>
        <div className="space-y-4">
          {data.strategy.map((item: any) => (
            <div key={item.id} className="flex items-center gap-4 bg-gray-50 p-4 rounded-xl border border-gray-200">
               <div className="w-6 h-6 rounded-full bg-white border border-gray-300 flex items-center justify-center text-[10px] font-bold text-gray-400">
                {item.id}
               </div>
               <span className="flex-1 font-bold text-gray-700 text-sm italic">{item.text}</span>
               <span className="font-black text-[#10B981] text-xs">+{item.impact}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
