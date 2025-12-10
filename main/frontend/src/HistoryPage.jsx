import React, { useEffect, useState, useRef } from 'react';
import { useAuth } from './context/AuthContext';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from './components/ui/dialog';
import { Button } from './components/ui/button';
import { useNavigate, Link } from 'react-router-dom';
import { FileText, Calendar, Activity, Printer, Download, UserSearch, Share2, Check, AlertTriangle, MessageSquare } from 'lucide-react';
import { PDFDownloadLink } from '@react-pdf/renderer';
import { useReactToPrint } from 'react-to-print';
import ReportPDF from './components/ReportPDF';
import PrintableReport from './components/PrintableReport';

const HistoryPage = () => {
  const { token, user, loading: authLoading } = useAuth();
  const navigate = useNavigate();
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedReport, setSelectedReport] = useState(null);
  const printRef = useRef();

  const handlePrint = () => {
    console.log("Print button clicked");
    if (printRef.current) {
      const printWindow = window.open('', '', 'height=600,width=800');
      printWindow.document.write('<html><head><title>Medical Report</title>');
      printWindow.document.write('<style>body { font-family: sans-serif; padding: 20px; }</style>');
      printWindow.document.write('</head><body>');
      printWindow.document.write(printRef.current.innerHTML);
      printWindow.document.write('</body></html>');
      printWindow.document.close();
      printWindow.print();
    } else {
      console.error("Print ref is null!");
      alert("Unable to print. Please try again.");
    }
  };


  const handleShare = async () => {
    if (navigator.share && selectedReport) {
      try {
        await navigator.share({
          title: 'MediVision Diagnostic Report',
          text: `Diagnosis: ${selectedReport.disease_detected} (Confidence: ${(selectedReport.confidence_score * 100).toFixed(1)}%)`,
          url: window.location.href
        });
      } catch (error) {
        console.log('Error sharing:', error);
      }
    } else {
      alert('Sharing is not supported on this browser/device.');
    }
  };

  useEffect(() => {
    // Wait for auth context to finish loading before checking authentication
    if (authLoading) {
      return;
    }

    // If auth loaded and no user, redirect to login
    if (!user || !token) {
      console.log("No authenticated user, redirecting to login");
      navigate('/login', { replace: true });
      return;
    }
    
    // User is authenticated, fetch history
    fetchHistory();
  }, [token, user, authLoading, navigate]);

  const fetchHistory = async () => {
    try {
      const response = await fetch('/history', {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        setReports(data);
      } else if (response.status === 401) {
        // Token expired or invalid, redirect to login
        console.log("Unauthorized, redirecting to login");
        navigate('/login', { replace: true });
      }
    } catch (error) {
      console.error("Failed to fetch history", error);
    } finally {
      setLoading(false);
    }
  };

  // Show loading while auth is being checked or history is being fetched
  if (authLoading || loading) {
    return (
      <div className="container mx-auto py-12 text-center">
        <div className="flex flex-col items-center gap-4">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
          <p>Loading history...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    // This shouldn't show due to redirect, but safety fallback
    return null;
  }

  return (
    <div className="container mx-auto py-8 px-4">
      <h1 className="text-3xl font-bold mb-8">Patient History</h1>
      
      {loading ? (
        <p>Loading history...</p>
      ) : reports.length === 0 ? (
        <div className="text-center py-12 bg-gray-50 dark:bg-gray-900 rounded-lg">
          <Activity className="h-12 w-12 mx-auto text-gray-400 mb-4" />
          <h3 className="text-xl font-medium mb-2">No Reports Found</h3>
          <p className="text-gray-500 mb-6">You haven't generated any diagnostic reports yet.</p>
          <Button asChild>
            <Link to="/upload">Start New Diagnosis</Link>
          </Button>
        </div>
      ) : (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {reports.map((report) => (
            <Card key={report.id} className="hover:shadow-lg transition-shadow">
              <CardHeader className="pb-2">
                <div className="flex justify-between items-start">
                  <CardTitle className="text-lg font-bold capitalize">
                    {report.modality} Scan
                  </CardTitle>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    report.disease_detected === 'Normal' || report.disease_detected === 'No Tumor' 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-red-100 text-red-800'
                  }`}>
                    {report.disease_detected}
                  </span>
                </div>
                <div className="text-sm text-gray-500 flex items-center gap-1">
                  <Calendar className="h-3 w-3" />
                  {new Date(report.created_at).toLocaleDateString()}
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Confidence:</span>
                    <span className="font-medium">{(report.confidence_score * 100).toFixed(1)}%</span>
                  </div>
                  <p className="text-sm text-gray-600 line-clamp-3 mt-2">
                    {report.report_text}
                  </p>
                  <div className="flex gap-2 mt-4">
                    <Button variant="outline" className="flex-1" onClick={() => setSelectedReport(report)}>
                        View Full Report
                    </Button>
                    <Button 
                        variant="secondary" 
                        size="icon"
                        onClick={() => navigate('/chat', { state: { reportContext: { report: report.report_text } } })}
                        title="Chat with AI"
                    >
                        <MessageSquare className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

      )}

      <Dialog open={!!selectedReport} onOpenChange={(open) => !open && setSelectedReport(null)}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="text-2xl font-bold flex items-center gap-2">
              {selectedReport?.modality} Scan Report
              {selectedReport && (
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                  selectedReport.disease_detected === 'Normal' || selectedReport.disease_detected === 'No Tumor' 
                    ? 'bg-green-100 text-green-800' 
                    : 'bg-red-100 text-red-800'
                }`}>
                  {selectedReport.disease_detected}
                </span>
              )}
            </DialogTitle>
            <DialogDescription>
              Generated on {selectedReport && new Date(selectedReport.created_at).toLocaleDateString()}
            </DialogDescription>
          </DialogHeader>
          
          {selectedReport && (
              <div className="space-y-6 mt-4">
                {/* Key Findings Section */}
                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-100 dark:border-blue-800">
                  <h4 className="text-sm font-semibold text-blue-800 dark:text-blue-300 mb-2 flex items-center gap-2">
                    <Activity className="h-4 w-4" /> Key Findings
                  </h4>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Primary Diagnosis</p>
                      <p className="text-lg font-bold text-gray-900 dark:text-gray-100">{selectedReport.disease_detected}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-600 dark:text-gray-400">AI Confidence</p>
                      <div className="flex items-center gap-1 justify-end">
                        <span className="text-lg font-bold text-blue-600">
                          {(selectedReport.confidence_score * 100).toFixed(1)}%
                        </span>
                        {selectedReport.confidence_score >= 0.85 && <Check className="h-4 w-4 text-green-500" />}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Detailed Report Section */}
                <div className="prose dark:prose-invert max-w-none">
                  <h4 className="text-lg font-semibold mb-2 flex items-center gap-2">
                    <FileText className="h-5 w-5 text-gray-500" /> Detailed Report
                  </h4>
                  <div className="whitespace-pre-wrap text-gray-700 dark:text-gray-300 leading-relaxed bg-gray-50 dark:bg-gray-800 p-4 rounded-md border">
                    {selectedReport.report_text}
                  </div>
                </div>
              </div>
          )}
          
          {selectedReport && (
            <div className="flex flex-wrap gap-3 mt-6 justify-end border-t pt-4">
              <Button variant="outline" onClick={() => {
                console.log("Print button clicked");
                handlePrint();
              }} className="flex items-center gap-2">
                <Printer size={16} /> Print
              </Button>

              <PDFDownloadLink
                document={<ReportPDF reportData={{
                  report: selectedReport.report_text,
                  confidence: (selectedReport.confidence_score * 100).toFixed(1),
                  diagnosis: selectedReport.disease_detected,
                  specialty: selectedReport.modality
                }} />}
                fileName={`Report_${selectedReport.id}.pdf`}
              >
                {({ loading }) => (
                  <Button variant="outline" disabled={loading} className="flex items-center gap-2">
                    <Download size={16} /> {loading ? 'Preparing...' : 'Download PDF'}
                  </Button>
                )}
              </PDFDownloadLink>

              <Button asChild className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700">
                <Link to="/search-doctor">
                  <UserSearch size={16} /> Find Doctor
                </Link>
              </Button>

              <Button variant="outline" onClick={handleShare} className="flex items-center gap-2">
                <Share2 size={16} /> Share
              </Button>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Hidden Printable Component - Always rendered but off-screen */}
      <div style={{ position: "fixed", top: "-10000px", left: "-10000px" }}>
        <PrintableReport 
          ref={printRef} 
          reportData={selectedReport ? {
            report: selectedReport.report_text,
            confidence: (selectedReport.confidence_score * 100).toFixed(1),
            symptoms: [], 
            diagnosis: selectedReport.disease_detected,
            specialty: selectedReport.modality
          } : {}} 
        />
      </div>
    </div>
  );
};

export default HistoryPage;
