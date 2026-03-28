"use client";

import React from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, Cell, Legend
} from 'recharts';

export default function OptimizationBarChart({ data }: { data: any }) {
  const chartData = [
    { name: 'Current', Tax: data?.current_annual_tax || 0, Savings: 0 },
    { name: 'Sentinel', Tax: data?.optimized_annual_tax || 0, Savings: data?.total_savings || 0 }
  ];

  const formatCurrency = (value: number) => `₹${(value / 1000).toFixed(0)}k`;

  return (
    <div className="bg-white p-6 rounded-xl border border-gray-100 shadow-sm h-full">
      <h3 className="font-serif font-bold text-lg mb-4 text-gray-900">Tax & Savings: Before vs After</h3>
      <div className="h-[250px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} stackOffset="none">
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f3f4f6" />
            <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fontSize: 12, fill: '#9ca3af'}} />
            <YAxis tickFormatter={formatCurrency} axisLine={false} tickLine={false} tick={{fontSize: 10, fill: '#9ca3af'}} />
            <Tooltip cursor={{fill: '#f9fafb'}} contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)'}} />
            <Legend verticalAlign="top" align="right" iconType="circle" />
            <Bar dataKey="Tax" stackId="a" fill="#1f2937" radius={[0, 0, 0, 0]} barSize={40} />
            <Bar dataKey="Savings" stackId="a" fill="#ED1C24" radius={[4, 4, 0, 0]} barSize={40} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
