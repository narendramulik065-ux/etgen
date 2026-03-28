"use client";

import React from 'react';
import { 
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend 
} from 'recharts';

export default function PortfolioDonut({ aum }: { aum: number }) {
  const data = [
    { name: 'Equity', value: 62 },
    { name: 'Debt', value: 25 },
    { name: 'Gold', value: 8 },
    { name: 'Cash', value: 5 },
  ];

  const COLORS = ['#ED1C24', '#1f2937', '#9ca3af', '#e5e7eb'];

  return (
    <div className="bg-white p-6 rounded-xl border border-gray-100 shadow-sm h-full flex flex-col items-center">
      <h3 className="font-serif font-bold text-lg mb-6 w-full text-left text-gray-900">Asset Allocation Breakdown</h3>
      <div className="h-[250px] w-full relative">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              paddingAngle={5}
              dataKey="value"
              stroke="none"
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend layout="vertical" verticalAlign="middle" align="right" iconType="circle" wrapperStyle={{ paddingLeft: '20px' }} />
          </PieChart>
        </ResponsiveContainer>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-center pointer-events-none" style={{ left: '42%' }}>
          <p className="text-[10px] uppercase font-bold text-gray-400">Total AUM</p>
          <p className="text-lg font-black text-gray-900">₹{(aum / 100000).toFixed(1)}L</p>
        </div>
      </div>
    </div>
  );
}
