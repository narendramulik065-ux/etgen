"use client";

import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

export default function LeakageMeter({ er, impact }: { er: number, impact: number }) {
  // Gauge logic: 0% to 3% range
  const value = er || 1.92;
  const data = [
    { value: value },
    { value: 3 - value }
  ];

  return (
    <div className="bg-white p-6 rounded-xl border border-gray-100 shadow-sm h-full">
      <h3 className="font-serif font-bold text-lg mb-4 text-gray-900">Portfolio Leakage Meter</h3>
      
      <div className="h-[180px] w-full relative">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="100%"
              startAngle={180}
              endAngle={0}
              innerRadius={70}
              outerRadius={90}
              paddingAngle={0}
              dataKey="value"
              stroke="none"
            >
              <Cell fill="#ED1C24" />
              <Cell fill="#f3f4f6" />
            </Pie>
          </PieChart>
        </ResponsiveContainer>
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 text-center">
          <p className="text-2xl font-black text-gray-900">{value}%</p>
          <p className="text-[10px] uppercase font-bold text-gray-400">Avg Expense</p>
        </div>
      </div>

      <div className="space-y-2 mt-4">
        <div className="flex justify-between text-[10px] font-bold">
          <span className="text-gray-400 uppercase">Industry Average</span>
          <span className="text-gray-900">1.35%</span>
        </div>
        <div className="flex justify-between text-[10px] font-bold">
          <span className="text-gray-400 uppercase">Optimal (Direct Plans)</span>
          <span className="text-green-600 font-black">0.85%</span>
        </div>
        <div className="pt-2 border-t border-gray-50 flex justify-between items-center">
            <span className="text-[10px] font-bold text-gray-400 uppercase">Your Excess Cost</span>
            <span className="text-sm font-black text-[#ED1C24]">1.07%</span>
        </div>
      </div>

      <div className="mt-6 p-4 bg-red-50 rounded-lg border border-red-100">
        <div className="flex items-center gap-2 mb-1">
            <span className="text-[10px] font-bold text-red-800 uppercase tracking-widest">10-Year Wealth Impact</span>
        </div>
        <p className="text-2xl font-black text-red-600">-₹{impact?.toLocaleString('en-IN')}</p>
        <p className="text-[9px] text-red-700/60 font-medium">Based on current corpus & 12% annual returns</p>
      </div>
    </div>
  );
}
