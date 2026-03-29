"use client";

import React, { useState, useEffect, useRef } from 'react';

interface Message {
  role: 'user' | 'ai';
  content: string;
}

export default function MentorChat({ financialContext }: { financialContext: any }) {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'ai', content: "I've analyzed your Form 16. How can I help you optimize this portfolio?" }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // 🏁 THE FIX: Points to your live Railway server instead of your laptop
  const API_BASE_URL = "https://etgen-production.up.railway.app";

  // Auto-scroll to latest message
  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      // Create the context object for the mentor
      const mentorContext = {
        salary: financialContext?.salary || 0,
        tax_paid: financialContext?.tax_paid || 0,
        savings: financialContext?.tax_leakage || 0
      };

      // Updated to use the live API URL
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: input,
          context: mentorContext
        }),
      });
      
      const data = await response.json();
      setMessages(prev => [...prev, { role: 'ai', content: data.reply }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'ai', content: "Connection error. Please check your backend." }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[500px] w-full bg-white border border-gray-200 shadow-xl rounded-xl overflow-hidden">
      {/* Header - ET Branded */}
      <div className="bg-[#ED1C24] p-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-white rounded-full flex items-center justify-center font-bold text-[#ED1C24]">ET</div>
          <span className="text-white font-serif font-bold text-lg">Money Mentor</span>
        </div>
        <span className="text-[10px] bg-white/20 text-white px-2 py-1 rounded-full uppercase tracking-widest font-bold">Sentinel AI</span>
      </div>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] p-3 rounded-lg text-sm shadow-sm ${
              msg.role === 'user' 
                ? 'bg-[#ED1C24] text-white rounded-br-none' 
                : 'bg-white text-gray-800 border border-gray-200 rounded-bl-none'
            }`}>
              {msg.content}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white p-3 rounded-lg shadow-sm animate-pulse text-gray-400 text-xs italic">
              Sentinel is calculating...
            </div>
          </div>
        )}
        <div ref={scrollRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 border-t border-gray-100 bg-white">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Ask about your tax savings..."
            className="flex-1 bg-gray-100 border-none rounded-lg px-4 py-2 text-sm focus:ring-2 focus:ring-[#ED1C24]"
          />
          <button 
            onClick={sendMessage}
            className="bg-[#ED1C24] text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors font-bold text-sm"
          >
            Ask
          </button>
        </div>
      </div>
    </div>
  );
}
