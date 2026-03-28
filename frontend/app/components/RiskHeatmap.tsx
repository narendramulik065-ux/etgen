"use client";

import React from 'react';

const funds = [
  { name: 'ICICI Pru Tech Fund', category: 'Equity', risk: 'High', overlap: '45%', er: '2.1%', rating: '★★★★☆' },
  { name: 'SBI Bluechip Fund', category: 'Equity', risk: 'Medium', overlap: '38%', er: '1.8%', rating: '★★★☆☆' },
  { name: 'HDFC Mid Cap Fund', category: 'Equity', risk: 'Very High', overlap: '12%', er: '2.3%', rating: '★★★★☆' },
  { name: 'ICICI Pru Balanced Adv', category: 'Hybrid', risk: 'Medium', overlap: '32%', er: '1.9%', rating: '★★★★★' },
  { name: 'SBI Magnum Gilt Fund', category: 'Debt', risk: 'Low', overlap: '5%', er: '1.2%', rating: '★★★☆☆' },
];

export default function RiskHeatmap() {
  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'High': return 'bg-red-50 text-red-600 border-red-100';
      case 'Very High': return 'bg-red-100 text-red-700 border-red-200 font-bold';
      case 'Medium': return 'bg-orange-50 text-orange-600 border-orange-100';
      case 'Low': return 'bg-green-50 text-green-600 border-green-100';
      default: return 'bg-gray-50 text-gray-600 border-gray-100';
    }
  };

  return (
    <div className="bg-white border border-gray-100 shadow-sm rounded-xl overflow-hidden">
      <div className="p-6 border-b border-gray-50 flex justify-between items-center">
        <div>
          <h3 className="font-serif font-bold text-lg text-gray-900">Fund-Level Risk Heatmap</h3>
          <p className="text-xs text-gray-400">Hover for detailed analysis per holding</p>
        </div>
        <div className="flex gap-4">
             <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-green-500"></span><span className="text-[10px] font-bold text-gray-500">Low Risk</span></div>
             <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-orange-400"></span><span className="text-[10px] font-bold text-gray-500">Medium</span></div>
             <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-red-500"></span><span className="text-[10px] font-bold text-gray-500">High</span></div>
        </div>
      </div>

      <table className="w-full text-left border-collapse">
        <thead className="bg-[#fafafa] text-[10px] font-bold uppercase tracking-widest text-gray-400 border-b border-gray-100">
          <tr>
            <th className="px-6 py-3 font-bold">Fund Name</th>
            <th className="px-6 py-3 font-bold">Category</th>
            <th className="px-6 py-3 font-bold text-center">Risk Level</th>
            <th className="px-6 py-3 font-bold text-center">Overlap</th>
            <th className="px-6 py-3 font-bold text-center">Expense Ratio</th>
            <th className="px-6 py-3 font-bold text-center">Rating</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-50">
          {funds.map((fund, i) => (
            <tr key={i} className="hover:bg-gray-50/50 transition-colors">
              <td className="px-6 py-4 text-xs font-bold text-gray-900">{fund.name}</td>
              <td className="px-6 py-4 text-xs text-gray-500">{fund.category}</td>
              <td className="px-6 py-4 text-center">
                <span className={`px-2.5 py-1 rounded-full text-[9px] uppercase border inline-block ${getRiskColor(fund.risk)}`}>
                  {fund.risk}
                </span>
              </td>
              <td className="px-6 py-4 text-xs text-gray-500 text-center font-bold tracking-tighter">{fund.overlap}</td>
              <td className="px-6 py-4 text-xs text-gray-500 text-center">{fund.er}</td>
              <td className="px-6 py-4 text-center text-[10px] text-[#ED1C24]">{fund.rating}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
