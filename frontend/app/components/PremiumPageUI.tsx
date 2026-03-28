"use client";

import React from 'react';

export default function PremiumPageUI() {
  return (
    <div className="space-y-8 animate-in fade-in zoom-in-95 duration-500">
      {/* Prime Hero Card */}
      <div className="bg-[#1A1A1A] rounded-3xl p-8 text-white relative overflow-hidden">
        <div className="relative z-10 max-w-2xl">
          <div className="flex items-center gap-2 mb-4">
             <span className="bg-[#ED1C24] text-white text-[10px] font-bold px-2 py-1 rounded">👑</span>
             <h3 className="text-xl font-serif font-bold">ET Prime Sentinel</h3>
          </div>
          <p className="text-gray-400 text-sm leading-relaxed mb-6">
            Unlock advanced AI-powered portfolio rebalancing, tax optimization strategies, and personalized wealth advisory insights trusted by high-net-worth individuals.
          </p>
          <div className="flex items-center gap-6">
            <div>
              <p className="text-[10px] text-gray-500 uppercase font-bold">Starting at</p>
              <p className="text-2xl font-serif font-black">₹999<span className="text-sm font-normal text-gray-400">/month</span></p>
            </div>
            <button className="bg-[#ED1C24] hover:bg-[#c1151b] transition-colors text-white px-8 py-3 rounded-xl font-bold text-sm shadow-lg shadow-red-900/20">
              Upgrade to Prime
            </button>
          </div>
        </div>
        {/* Decorative background element */}
        <div className="absolute top-0 right-0 w-64 h-64 bg-red-600/10 blur-[100px] rounded-full"></div>
      </div>

      {/* Feature Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <FeatureCard 
          icon="📈" 
          title="AI Portfolio Rebalancing" 
          desc="Get quarterly AI-driven rebalancing recommendations optimized for tax efficiency and market conditions."
          items={["Tax-loss harvesting opportunities", "Automated alerts for rebalancing triggers"]}
        />
        <FeatureCard 
          icon="🛡️" 
          title="Advanced Tax Optimization" 
          desc="Maximize post-tax returns with personalized tax-saving strategies across all income sources."
          items={["Annual tax projection & planning", "Section 80C to 80U optimization"]}
        />
        <FeatureCard 
          icon="🎯" 
          title="Goal-Based Planning Pro" 
          desc="Create and track multiple financial goals with AI-powered milestone predictions."
          items={["Unlimited goal tracking", "Monte Carlo simulations"]}
        />
        <FeatureCard 
          icon="⚡" 
          title="Priority Support & Insights" 
          desc="Get priority access to ET's financial experts and exclusive market insights."
          items={["24/7 priority email support", "Monthly expert webinars"]}
        />
      </div>
    </div>
  );
}

function FeatureCard({ icon, title, desc, items }: { icon: string, title: string, desc: string, items: string[] }) {
  return (
    <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-sm hover:shadow-md transition-shadow">
      <div className="text-2xl mb-4">{icon}</div>
      <h4 className="font-bold text-gray-900 mb-2">{title}</h4>
      <p className="text-xs text-gray-500 mb-4">{desc}</p>
      <ul className="space-y-2">
        {items.map((item, i) => (
          <li key={i} className="flex items-center gap-2 text-[11px] text-gray-700 font-medium">
            <span className="text-[#10B981]">✓</span> {item}
          </li>
        ))}
      </ul>
    </div>
  );
}
