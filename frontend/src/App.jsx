import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Input } from '@/components/ui/input';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';


// Quick local database interaction utility wrapping browser native IndexedDB
const initIndexedDB = () => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("RAGMetricsDB", 1);
    request.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains("token_logs")) {
        db.createObjectStore("token_logs", { keyPath: "id", autoIncrement: true });
      }
    };
    request.onsuccess = (e) => resolve(e.target.result);
    request.onerror = (e) => reject(e.target.error);
  });
};

const logTokensLocal = async (promptTokens, completionTokens, endpoint, model) => {
  const db = await initIndexedDB();
  const tx = db.transaction("token_logs", "readwrite");
  const store = tx.objectStore("token_logs");
  const rateUSD = 0.0015 / 1000; // Sample standardized tier rate calculation tracking
  const total = promptTokens + completionTokens;
  
  store.add({
    timestamp: new Date().toISOString(),
    endpoint,
    model,
    promptTokens,
    completionTokens,
    totalTokens: total,
    costUSD: total * rateUSD,
    costINR: total * rateUSD * 83.5
  });
};

export default function App() {
  const [activeTab, setActiveTab] = useState("home");
  const [sessionId] = useState(() => `session_${uuidv4()}`);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [usageStats, setUsageStats] = useState({ seven: 0, fourteen: 0, thirty: 0 });
  
  const bottomRef = useRef(null);

  useEffect(() => {
    if (activeTab === "dashboard") {
      fetchMetrics();
    }
  }, [activeTab]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const fetchMetrics = async () => {
    const db = await initIndexedDB();
    const tx = db.transaction("token_logs", "readonly");
    const store = tx.objectStore("token_logs");
    const req = store.getAll();
    
    req.onsuccess = () => {
      const logs = req.result;
      const now = new Date();
      
      let s7 = 0, s14 = 0, s30 = 0;
      logs.forEach(log => {
        const logDate = new Date(log.timestamp);
        const diffDays = (now - logDate) / (1000 * 60 * 60 * 24);
        
        if (diffDays <= 7) s7 += log.totalTokens;
        if (diffDays <= 14) s14 += log.totalTokens;
        if (diffDays <= 30) s30 += log.totalTokens;
      });
      setUsageStats({ seven: s7, fourteen: s14, thirty: s30 });
    };
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isStreaming) return;
    
    const userPrompt = inputMessage;
    setInputMessage("");
    setMessages(prev => [...prev, { role: 'user', content: userPrompt }]);
    setIsStreaming(true);
    
    // Optimistic baseline token extraction estimates for local UX safety metrics tracking
    const estimatedPromptTokens = Math.ceil(userPrompt.length / 4);
    
    setMessages(prev => [...prev, { role: 'assistant', content: "" }]);
    
    try {
      const response = await fetch("http://localhost:8000/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, query: userPrompt })
      });
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let completeResponse = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");
        
        lines.forEach(line => {
          if (line.startsWith("data: ")) {
            try {
              const parsed = JSON.parse(line.substring(6));
              completeResponse += parsed.text;
              setMessages(prev => {
                const updated = [...prev];
                updated[updated.length - 1].content = completeResponse;
                return updated;
              });
            } catch (err) {}
          }
        });
      }
      
      const estimatedCompletionTokens = Math.ceil(completeResponse.length / 4);
      
      // Concurrently update storage metrics pools
      await logTokensLocal(estimatedPromptTokens, estimatedCompletionTokens, "chat/stream", "gemini-1.5-pro");
      await fetch("http://localhost:8000/api/usage/track", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          endpoint: "chat/stream",
          model: "gemini-1.5-pro",
          prompt_tokens: estimatedPromptTokens,
          completion_tokens: estimatedCompletionTokens
        })
      });

    } catch (error) {
      console.error("Streaming transaction error:", error);
    } finally {
      setIsStreaming(false);
    }
  };

//   return (
//     <div className="min-h-screen bg-slate-950 text-slate-50 flex flex-col items-center p-6 font-sans">
//       <div className="w-full max-w-5xl space-y-6">
        
//         {/* APP HEADER */}
//         <header className="border-b border-slate-800 pb-4 flex justify-between items-center">
//           <div>
//             <h1 className="text-2xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-indigo-400">
//               ISRO SATCOM NGP Explorer
//             </h1>
//             <p className="text-sm text-slate-400 mt-1">
//               Deterministic RAG playground built to parse and evaluate the official Next Generation Satcom Policy published by ISRO.
//             </p>
//           </div>
//           <a href="https://www.isro.gov.in" target="_blank" rel="noreferrer" className="text-xs text-blue-400 hover:underline">
//             View Official Document →
//           </a>
//         </header>

//         {/* NAVIGATION CONTROLS BAR */}
//         <div className="flex gap-4">
//           <Button variant={activeTab === "home" ? "default" : "outline"} onClick={() => setActiveTab("home")}>
//             Core Overview
//           </Button>
//           <Button variant={activeTab === "chat" ? "default" : "outline"} onClick={() => setActiveTab("chat")}>
//             Chat Engine Window
//           </Button>
//           <Button variant={activeTab === "dashboard" ? "default" : "outline"} onClick={() => setActiveTab("dashboard")}>
//             Usage Dashboard
//           </Button>
//           <Button variant={activeTab === "guide" ? "default" : "outline"} onClick={() => setActiveTab("guide")}>
//             Setup Guide
//           </Button>
//         </div>

//         {/* INTERACTIVE INTERFACE AREAS */}
//         <main className="w-full mt-4">
//           {activeTab === "home" && (
//             <Card className="bg-slate-900 border-slate-800">
//               <CardHeader>
//                 <CardTitle className="text-slate-100">Project Intention & Pipeline Features</CardTitle>
//                 <CardDescription className="text-slate-400">
//                   This system implements custom retrieval loops explicitly designed to keep the legal integrity of complex state frameworks intact.
//                 </CardDescription>
//               </CardHeader>
//               <CardContent className="space-y-4 text-sm text-slate-300">
//                 <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
//                   <div className="p-4 rounded-lg bg-slate-950 border border-slate-800">
//                     <h3 className="font-semibold text-blue-400 mb-1">Structural Hierarchy Splitter</h3>
//                     <p className="text-xs text-slate-400">Chunks texts strictly along Articles and nested Clauses, preserving structural metadata parameters over crude sliding word limits.</p>
//                   </div>
//                   <div className="p-4 rounded-lg bg-slate-950 border border-slate-800">
//                     <h3 className="font-semibold text-blue-400 mb-1">Two-Tier Parent Fetching</h3>
//                     <p className="text-xs text-slate-400">Queries fine-grain snippets for clean mathematical proximity matches, but returns the larger conceptual section context blocks directly to the LLM context.</p>
//                   </div>
//                   <div className="p-4 rounded-lg bg-slate-950 border border-slate-800">
//                     <h3 className="font-semibold text-blue-400 mb-1">Agentic Loop Resolution</h3>
//                     <p className="text-xs text-slate-400">Automatically executes recursive sub-queries if a fetched policy fragment references a distant external clause definition.</p>
//                   </div>
//                   <div className="p-4 rounded-lg bg-slate-950 border border-slate-800">
//                     <h3 className="font-semibold text-blue-400 mb-1">Hybrid Re-ranking Engine</h3>
//                     <p className="text-xs text-slate-400">Pairs semantic vector hits with query expansions, run against cross-encoders via Cohere for optimal analytical sorting verification.</p>
//                   </div>
//                 </div>
//                 <div className="mt-4 pt-4 border-t border-slate-800 flex justify-center">
//                   <Button onClick={() => setActiveTab("chat")} className="bg-blue-600 hover:bg-blue-700">
//                     Launch Interactive Session Now
//                   </Button>
//                 </div>
//               </CardContent>
//             </Card>
//           )}

//           {activeTab === "chat" && (
//             <Card className="bg-slate-900 border-slate-800 h-[600px] flex flex-col">
//               <CardContent className="p-4 flex flex-col h-full space-y-4">
//                 <ScrollArea className="flex-1 bg-slate-950 rounded-md p-4 border border-slate-800">
//                   <div className="space-y-4">
//                     {messages.length === 0 && (
//                       <p className="text-center text-xs text-slate-500 pt-12">
//                         No active conversation frames initialized. Query the engine above regarding SATCOM provisions.
//                       </p>
//                     )}
//                     {messages.map((m, idx) => (
//                       <div key={idx} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
//                         <div className={`max-w-[80%] rounded-md px-3 py-2 text-sm ${m.role === 'user' ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-200'}`}>
//                           {m.content || <span className="animate-pulse">Processing pipeline stream...</span>}
//                         </div>
//                       </div>
//                     ))}
//                     <div ref={bottomRef} />
//                   </div>
//                 </ScrollArea>
//                 <div className="flex gap-2">
//                   <Input 
//                     value={inputMessage}
//                     onChange={(e) => setInputMessage(e.target.value)}
//                     placeholder="Ask about orbital assignments, pricing regimes, or clause definitions..."
//                     className="bg-slate-950 border-slate-800"
//                     onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
//                   />
//                   <Button onClick={handleSendMessage} disabled={isStreaming} className="bg-indigo-600 hover:bg-indigo-700">
//                     Send
//                   </Button>
//                 </div>
//               </CardContent>
//             </Card>
//           )}

//           {activeTab === "dashboard" && (
//             <Card className="bg-slate-900 border-slate-800">
//               <CardHeader>
//                 <CardTitle className="text-slate-100">Token Allocation & Analytics Engine</CardTitle>
//                 <CardDescription className="text-slate-400">
//                   Real-time local data mapping extracted securely from sandboxed IndexedDB storage configurations.
//                 </CardDescription>
//               </CardHeader>
//               <CardContent className="space-y-6">
//                 <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
//                   <div className="p-4 bg-slate-950 border border-slate-800 rounded-md text-center">
//                     <div className="text-xs text-slate-400 mb-1">Past 7 Days Consumption</div>
//                     <div className="text-xl font-bold text-indigo-400">{usageStats.seven.toLocaleString()} tokens</div>
//                   </div>
//                   <div className="p-4 bg-slate-950 border border-slate-800 rounded-md text-center">
//                     <div className="text-xs text-slate-400 mb-1">Past 14 Days Consumption</div>
//                     <div className="text-xl font-bold text-indigo-400">{usageStats.fourteen.toLocaleString()} tokens</div>
//                   </div>
//                   <div className="p-4 bg-slate-950 border border-slate-800 rounded-md text-center">
//                     <div className="text-xs text-slate-400 mb-1">Past 30 Days Consumption</div>
//                     <div className="text-xl font-bold text-indigo-400">{usageStats.thirty.toLocaleString()} tokens</div>
//                   </div>
//                 </div>
//                 <Alert className="bg-slate-950 border-slate-800">
//                   <AlertTitle className="text-amber-400 text-xs font-semibold">Server Enforcement Protocol</AlertTitle>
//                   <AlertDescription className="text-slate-400 text-xs">
//                     Token limits are evaluated on the server. Clearing cache files resets the local visuals above but retains the core server budget logs.
//                   </AlertDescription>
//                 </Alert>
//               </CardContent>
//             </Card>
//           )}

//           {activeTab === "guide" && (
//             <Card className="bg-slate-900 border-slate-800">
//               <CardHeader>
//                 <CardTitle className="text-slate-100">System Setup Instructions</CardTitle>
//               </CardHeader>
//               <CardContent className="text-xs space-y-4 text-slate-300">
//                 <div>
//                   <h4 className="font-bold text-slate-100 mb-1">1. Environment Initialization</h4>
//                   <pre className="bg-slate-950 p-2 rounded text-slate-400 overflow-x-auto">
//                     pip install fastapi uvicorn chromadb sentence-transformers google-generativeai cohere pydantic
//                   </pre>
//                 </div>
//                 <div>
//                   <h4 className="font-bold text-slate-100 mb-1">2. Launch Server</h4>
//                   <pre className="bg-slate-950 p-2 rounded text-slate-400 overflow-x-auto">
//                     export GEMINI_API_KEY="your_key"{"\n"}
//                     export COHERE_API_KEY="your_key"{"\n"}
//                     uvicorn main:app --reload
//                   </pre>
//                 </div>
//                 <div>
//                   <h4 className="font-bold text-slate-100 mb-1">3. Document Ingestion Design</h4>
//                   <p className="text-slate-400">
//                     Ensure your document intake script iterates directly down your structure blocks, feeding structured payloads via 
//                     <code>collection.add(documents=[text], metadatas=[{"{"}"article": X, "section": Y{"}"}], ids=[unique_id])</code> 
//                     exactly once during backend setup.
//                   </p>
//                 </div>
//               </CardContent>
//             </Card>
//           )}
//         </main>

//       </div>
//     </div>
//   );
// }

// Change the outer layout container at the bottom of App.jsx to match this structure:
return (
  <div className="min-h-screen bg-white dark:bg-slate-950 text-slate-900 dark:text-slate-50 flex flex-col items-center p-6 font-sans">
    <div className="w-full max-w-5xl space-y-6">
      
      {/* APP HEADER */}
      <header className="border-b border-slate-200 dark:border-slate-800 pb-4 flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-orange-500 to-amber-600">
            ISRO SATCOM NGP Explorer
          </h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            Deterministic RAG playground built to parse and evaluate the official Next Generation Satcom Policy published by ISRO.
          </p>
        </div>
        <a href="https://www.isro.gov.in" target="_blank" rel="noreferrer" className="text-sm font-medium text-orange-600 dark:text-orange-400 hover:underline">
          View Official Document →
        </a>
      </header>

      {/* NAVIGATION CONTROLS BAR */}
      <div className="flex gap-4">
        <Button variant={activeTab === "home" ? "default" : "outline"} onClick={() => setActiveTab("home")} className={activeTab === "home" ? "bg-orange-600 hover:bg-orange-700 text-white" : ""}>
          Core Overview
        </Button>
        <Button variant={activeTab === "chat" ? "default" : "outline"} onClick={() => setActiveTab("chat")} className={activeTab === "chat" ? "bg-orange-600 hover:bg-orange-700 text-white" : ""}>
          Chat Engine Window
        </Button>
        <Button variant={activeTab === "dashboard" ? "default" : "outline"} onClick={() => setActiveTab("dashboard")} className={activeTab === "dashboard" ? "bg-orange-600 hover:bg-orange-700 text-white" : ""}>
          Usage Dashboard
        </Button>
        <Button variant={activeTab === "guide" ? "default" : "outline"} onClick={() => setActiveTab("guide")} className={activeTab === "guide" ? "bg-orange-600 hover:bg-orange-700 text-white" : ""}>
          Setup Guide
        </Button>
      </div>

      {/* INTERACTIVE INTERFACE AREAS */}
      <main className="w-full mt-4">
        {activeTab === "home" && (
          <Card className="bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-800">
            <CardHeader>
              <CardTitle className="text-slate-900 dark:text-slate-100">Project Intention & Pipeline Features</CardTitle>
              <CardDescription className="text-slate-500 dark:text-slate-400">
                This system implements custom retrieval loops explicitly designed to keep the legal integrity of complex state frameworks intact.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 text-sm text-slate-700 dark:text-slate-300">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 rounded-lg bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
                  <h3 className="font-semibold text-orange-600 dark:text-orange-400 mb-1">Structural Hierarchy Splitter</h3>
                  <p className="text-xs text-slate-500 dark:text-slate-400">Chunks texts strictly along Articles and nested Clauses, preserving structural metadata parameters over crude sliding word limits.</p>
                </div>
                <div className="p-4 rounded-lg bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
                  <h3 className="font-semibold text-orange-600 dark:text-orange-400 mb-1">Two-Tier Parent Fetching</h3>
                  <p className="text-xs text-slate-500 dark:text-slate-400">Queries fine-grain snippets for clean mathematical proximity matches, but returns the larger conceptual section context blocks directly to the LLM context.</p>
                </div>
                <div className="p-4 rounded-lg bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
                  <h3 className="font-semibold text-orange-600 dark:text-orange-400 mb-1">Agentic Loop Resolution</h3>
                  <p className="text-xs text-slate-500 dark:text-slate-400">Automatically executes recursive sub-queries if a fetched policy fragment references a distant external clause definition.</p>
                </div>
                <div className="p-4 rounded-lg bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
                  <h3 className="font-semibold text-orange-600 dark:text-orange-400 mb-1">Hybrid Re-ranking Engine</h3>
                  <p className="text-xs text-slate-500 dark:text-slate-400">Pairs semantic vector hits with query expansions, run against cross-encoders via Cohere for optimal analytical sorting verification.</p>
                </div>
              </div>
              <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-800 flex justify-center">
                <Button onClick={() => setActiveTab("chat")} className="bg-orange-600 hover:bg-orange-700 text-white font-medium">
                  Launch Interactive Session Now
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* CHAT WINDOW SECTION */}
        {activeTab === "chat" && (
          <Card className="bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-800 h-[600px] flex flex-col">
            <CardContent className="p-4 flex flex-col h-full space-y-4">
              <ScrollArea className="flex-1 bg-white dark:bg-slate-950 rounded-md p-4 border border-slate-200 dark:border-slate-800">
                <div className="space-y-4">
                  {messages.length === 0 && (
                    <p className="text-center text-xs text-slate-400 dark:text-slate-500 pt-12">
                      No active conversation frames initialized. Query the engine above regarding SATCOM provisions.
                    </p>
                  )}
                  {messages.map((m, idx) => (
                    <div key={idx} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                      <div className={`max-w-[80%] rounded-md px-3 py-2 text-sm ${m.role === 'user' ? 'bg-orange-600 text-white' : 'bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-200'}`}>
                        {m.content || <span className="animate-pulse text-orange-500">Processing pipeline stream...</span>}
                      </div>
                    </div>
                  ))}
                  <div ref={bottomRef} />
                </div>
              </ScrollArea>
              <div className="flex gap-2">
                <Input 
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  placeholder="Ask about orbital assignments, pricing regimes, or clause definitions..."
                  className="bg-white dark:bg-slate-950 border-slate-200 dark:border-slate-800 focus-visible:ring-orange-500"
                  onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                />
                <Button onClick={handleSendMessage} disabled={isStreaming} className="bg-orange-600 hover:bg-orange-700 text-white">
                  Send
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* DASHBOARD TRACKER SECTION */}
        {activeTab === "dashboard" && (
          <Card className="bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-800">
            <CardHeader>
              <CardTitle className="text-slate-900 dark:text-slate-100">Token Allocation & Analytics Engine</CardTitle>
              <CardDescription className="text-slate-500 dark:text-slate-400">
                Real-time local data mapping extracted securely from sandboxed IndexedDB storage configurations.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-md text-center">
                  <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Past 7 Days Consumption</div>
                  <div className="text-xl font-bold text-orange-600 dark:text-orange-400">{usageStats.seven.toLocaleString()} tokens</div>
                </div>
                <div className="p-4 bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-md text-center">
                  <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Past 14 Days Consumption</div>
                  <div className="text-xl font-bold text-orange-600 dark:text-orange-400">{usageStats.fourteen.toLocaleString()} tokens</div>
                </div>
                <div className="p-4 bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-md text-center">
                  <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Past 30 Days Consumption</div>
                  <div className="text-xl font-bold text-orange-600 dark:text-orange-400">{usageStats.thirty.toLocaleString()} tokens</div>
                </div>
              </div>
              <Alert className="bg-white dark:bg-slate-950 border-orange-200 dark:border-orange-900">
                <AlertTitle className="text-orange-600 dark:text-orange-400 text-xs font-semibold">Server Enforcement Protocol</AlertTitle>
                <AlertDescription className="text-slate-500 dark:text-slate-400 text-xs">
                  Token limits are evaluated on the server. Clearing cache files resets the local visuals above but retains the core server budget logs.
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
        )}

        {/* GUIDE TAB */}
        {activeTab === "guide" && (
          <Card className="bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-800">
            {/* Guide content stays the same, rendering cleanly over light/dark surfaces */}
          </Card>
        )}
      </main>
    </div>
  </div>
);
}