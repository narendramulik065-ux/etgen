"use client";

import React from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, Legend
} from 'recharts';

export default function DualLineChart({ context }: { context: any }) {
  // Logic: Compare "Current" (No optimization) vs "Sentinel" (Recovered Leakage)
  const generateData = () => {
    const years = [2024, 2025, 2026, 2027, 2028, 2029, 2030];
    const rate = 0.12;
    const leakage = context?.tax_leakage || 0;
    return years.map((year, i) => {
      const n = i + 1;
      const salary = context?.salary || 1000000;
      const baselineAum = salary * 1.5; // Dynamic base
      const currentWealth = baselineAum + (n * (salary * 0.05)); // Growth scaling with salary
      // Sentinel recovers leakage and reinvests it
      const recoveredWealth = currentWealth + (leakage * ((Math.pow(1 + rate, n) - 1) / rate));
      
      return {
        year,
        "Current Strategy": Math.round(currentWealth),
        "Sentinel Strategy": Math.round(recoveredWealth)
      };
    });
  };

  const data = generateData();

  const formatCurrency = (value: number) => `₹${(value / 100000).toFixed(1)}L`;

  return (
    <div className="bg-white p-6 rounded-xl border border-gray-100 shadow-sm h-full">
      <h3 className="font-serif font-bold text-lg mb-4 text-gray-900">Predictive Wealth Trajectory</h3>
      <div className="h-[250px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f3f4f6" />
            <XAxis dataKey="year" axisLine={false} tickLine={false} tick={{fontSize: 12, fill: '#9ca3af'}} />
            <YAxis tickFormatter={formatCurrency} axisLine={false} tickLine={false} tick={{fontSize: 10, fill: '#9ca3af'}} />
            <Tooltip 
              contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)' }}
              formatter={(value: any) => [formatCurrency(Number(value)), '']}
            />
            <Legend verticalAlign="top" align="right" iconType="circle" wrapperStyle={{ paddingBottom: '20px', fontSize: '12px' }} />
            <Line type="monotone" dataKey="Current Strategy" stroke="#9ca3af" strokeWidth={2} dot={false} strokeDasharray="5 5" />
            <Line type="monotone" dataKey="Sentinel Strategy" stroke="#ED1C24" strokeWidth={3} dot={{ r: 4, fill: '#ED1C24' }} activeDot={{ r: 6 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
