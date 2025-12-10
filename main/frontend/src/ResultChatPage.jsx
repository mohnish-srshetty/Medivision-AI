import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Button } from "./components/ui/button";
import { Input } from "./components/ui/input";
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from './components/ui/card';
import { useAuth } from './context/AuthContext';
import { Send, ArrowLeft, Bot, User, FileText, ChevronDown } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "./components/ui/dropdown-menu";

const ResultChatPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { token, user } = useAuth();

  // Initial context from navigation (if any)
  const initialContext = location.state?.reportContext || null;

  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I can help you understand your medical report. What would you like to know?' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [reports, setReports] = useState([]);
  const [currentContext, setCurrentContext] = useState(initialContext);
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Fetch history if no initial context is provided
  useEffect(() => {
    if (!initialContext && token) {
      fetchHistory();
    }
  }, [initialContext, token]);

  const fetchHistory = async () => {
    try {
      const response = await fetch('/history', {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (response.ok) {
        const data = await response.json();
        setReports(data);
      }
    } catch (error) {
      console.error("Failed to fetch history", error);
    }
  };

  const handleReportSelect = (report) => {
    setCurrentContext(report);

    // Add system message acknowledging the selection
    const disease = report.disease_detected || "Unknown Condition";
    const date = new Date(report.created_at).toLocaleDateString();

    setMessages(prev => [
      ...prev,
      {
        role: 'assistant',
        content: `I've switched context to your **${report.modality}** report from **${date}** (Diagnosis: ${disease}). \n\nHere is the summary:\n${report.report_text.substring(0, 200)}...\n\nWhat specific questions do you have about this report?`
      }
    ]);
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post('/chat_with_report/', {
        message: input,
        report_context: currentContext ? {
          report: currentContext.report_text,
          disease: currentContext.disease_detected
        } : null
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });

      const aiMessage = { role: 'assistant', content: response.data.response };
      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error("Chat error:", error);
      setMessages(prev => [...prev, { role: 'assistant', content: "I'm sorry, I encountered an error. Please try again." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 p-4 md:p-8 flex justify-center">
      <Card className="w-full max-w-4xl h-[80vh] flex flex-col shadow-lg dark:border-slate-800">
        <CardHeader className="border-b bg-white dark:bg-slate-900 dark:border-slate-800 rounded-t-lg flex flex-row items-center justify-between sticky top-0 z-10">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" onClick={() => navigate(-1)}>
              <ArrowLeft className="h-5 w-5" />
            </Button>
            <div className="flex flex-col">
              <CardTitle className="text-xl text-blue-700 dark:text-blue-400 flex items-center gap-2">
                <Bot className="h-6 w-6" /> AI Medical Assistant
              </CardTitle>
              {currentContext && (
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  Discussing: {currentContext.disease_detected || "Medical Report"}
                </span>
              )}
            </div>
          </div>

          {/* Report Selection Dropdown (only if not fixed context) */}
          {!initialContext && reports.length > 0 && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" className="flex items-center gap-2">
                  <FileText className="h-4 w-4" />
                  {currentContext ? "Switch Report" : "Select Report"}
                  <ChevronDown className="h-4 w-4 opacity-50" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-64 max-h-80 overflow-y-auto">
                {reports.map((report) => (
                  <DropdownMenuItem key={report.id} onClick={() => handleReportSelect(report)}>
                    <div className="flex flex-col">
                      <span className="font-medium">{report.disease_detected}</span>
                      <span className="text-xs text-slate-500">
                        {report.modality} • {new Date(report.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          )}
        </CardHeader>

        <CardContent className="flex-1 overflow-hidden p-0 bg-slate-50 dark:bg-slate-950 relative">
          <div className="h-full p-4 overflow-y-auto">
            <div className="space-y-4 pb-4">
              {messages.map((msg, index) => (
                <div
                  key={index}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] p-3 rounded-2xl shadow-sm ${msg.role === 'user'
                        ? 'bg-blue-600 text-white rounded-br-none'
                        : 'bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-100 border border-slate-200 dark:border-slate-800 rounded-bl-none'
                      }`}
                  >
                    <div className="flex items-center gap-2 mb-1 opacity-70 text-xs">
                      {msg.role === 'user' ? <User size={12} /> : <Bot size={12} />}
                      <span>{msg.role === 'user' ? 'You' : 'AI Assistant'}</span>
                    </div>
                    <p className="text-sm md:text-base leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                  </div>
                </div>
              ))}
              {loading && (
                <div className="flex justify-start">
                  <div className="bg-white dark:bg-slate-900 p-4 rounded-2xl rounded-bl-none shadow-sm border border-slate-200 dark:border-slate-800">
                    <div className="flex gap-1">
                      <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                      <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                      <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={scrollRef} />
            </div>
          </div>
        </CardContent>

        <CardFooter className="bg-white dark:bg-slate-900 border-t dark:border-slate-800 p-4">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
            className="flex w-full items-center gap-2"
          >
            <Input
              type="text"
              placeholder={currentContext ? "Ask about this report..." : "Select a report or ask general questions..."}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading}
              className="flex-1 bg-transparent dark:text-white"
              autoComplete="off"
            />
            <Button type="submit" disabled={loading || !input.trim()} size="icon" className="bg-blue-600 hover:bg-blue-700">
              <Send className="h-4 w-4" />
            </Button>
          </form>
        </CardFooter>
      </Card>
    </div>
  );
};

export default ResultChatPage;
