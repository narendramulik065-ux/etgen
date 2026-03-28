"use client";

import React from 'react';

export default function HouseholdAlignment({ score }: { score: number }) {
  // SVG Circular Gauge Logic
  const radius = 35;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;

  return (
    <div className="bg-white p-6 rounded-xl border border-gray-100 shadow-sm h-full flex flex-col justify-between">
      <h3 className="font-serif font-bold text-lg text-gray-900">Household Alignment</h3>
      <p className="text-[10px] text-gray-400 font-bold uppercase tracking-widest -mt-1 mb-4">Financial synergy between partners</p>
      
      <div className="relative flex items-center justify-center py-4">
        <svg className="w-24 h-24 transform -rotate-90">
          {/* Background Circle */}
          <circle
            cx="48"
            cy="48"
            r={radius}
            stroke="currentColor"
            strokeWidth="8"
            fill="transparent"
            className="text-gray-100"
          />
          {/* Progress Circle */}
          <circle
            cx="48"
            cy="48"
            r={radius}
            stroke="currentColor"
            strokeWidth="8"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            fill="transparent"
            className="text-[#ED1C24] transition-all duration-1000 ease-out"
          />
        </svg>
        <span className="absolute text-xl font-black font-serif text-gray-900">{score}%</span>
      </div>
    </div>
  );
}
