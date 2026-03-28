"use client";

import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

// 🎯 FIXED: Now accepts pre-calculated data and the gain label
export default function WealthProjectionChart({ chartData, ultimateGain }: { chartData: number[], ultimateGain: string }) {
  
  // Maps the backend array [Year 0, 5, 10, 15, 20] to the chart format
  const years = [0, 5, 10, 15, 20];
  const data = chartData ? chartData.map((val, i) => ({
    name: `Year ${years[i]}`,
    wealth: val
  })) : [5, 10, 15, 20].map(y => ({ name: `Year ${y}`, wealth: 0 }));

  return (
    <div className="bg-white p-6 rounded-xl border border-gray-100 shadow-sm h-full flex flex-col justify-between">
      <div>
        <h3 className="font-serif font-bold text-lg text-gray-900 leading-tight">20-Year Wealth Trajectory</h3>
        <p className="text-[10px] text-gray-400 font-bold uppercase tracking-widest">Projected growth @ 12% CAGR</p>
      </div>

      <div className="h-[180px] mt-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fontSize: 9, fill: '#9ca3af', fontWeight: 'bold'}} />
            <YAxis hide />
            <Tooltip 
              cursor={{fill: 'transparent'}}
              formatter={(value: any) => [`₹${(Number(value)/100000).toFixed(1)}L`, 'Wealth']}
            />
            <Bar dataKey="wealth" radius={[4, 4, 0, 0]} barSize={32}>
              {data.map((_, i) => (
                <Cell key={i} fill={i === 4 ? '#ED1C24' : '#1F2937'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="flex justify-between items-end mt-2 leading-none">
        <div className="flex items-center gap-1.5">
           <div className="w-2 h-2 rounded-full bg-[#ED1C24]"></div>
           <span className="text-[8px] font-bold text-gray-400 uppercase tracking-widest">Ultimate Gain</span>
        </div>
        {/* 🎯 FIXED: Label is now dynamic */}
        <span className="text-xl font-black font-serif text-[#ED1C24] tracking-tighter">
            {ultimateGain || "₹0L"}
        </span>
      </div>
    </div>
  );
}
