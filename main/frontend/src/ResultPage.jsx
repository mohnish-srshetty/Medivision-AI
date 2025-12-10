import React, { useEffect, useState } from 'react';
import ReportCard from './ReportCard';
import { useParams, useLocation, useNavigate } from 'react-router-dom';

const BASE_API_URL = '';

const ResultsPage = ({ processedData: propProcessedData, selectedImageType: propSelectedImageType }) => {
  const { cleanType } = useParams(); // e.g., 'xray', 'ct', etc.
  const location = useLocation();
  const navigate = useNavigate();

  // Try to get the passed state from navigation, or use props, or use global state if we had context
  const { selectedImageType: stateSelectedImageType, processedData: stateProcessedData } = location.state || {};
  
  const processedData = stateProcessedData || propProcessedData;
  const selectedImageType = stateSelectedImageType || propSelectedImageType;

  const [reportData, setReportData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // If we have processedData from navigation, use it directly
    if (processedData) {
      // Construct your reportData using processedData from UploadPage
      const predictionData = processedData.predictions || [];
      const reportText = processedData.report || '';
      const disease = processedData.disease || '';
      const symptoms = processedData.symptoms || [];
      
      // DEBUG: Check response structure
      console.log('=== FULL RESPONSE DEBUG ===');
      console.log('processedData:', processedData);
      console.log('processedData keys:', Object.keys(processedData));
      console.log('confidence_score:', processedData.confidence_score);
      console.log('findings:', processedData.findings);
      console.log('recommendations:', processedData.recommendations);
      console.log('suggested_tests:', processedData.suggested_tests);
      console.log('=========================');

      // Find the top prediction for confidence and diagnosis
      const sorted = Array.isArray(predictionData) && predictionData.length
        ? [...predictionData].sort((a, b) => b[1] - a[1])
        : [];

      const topK = sorted.slice(0, 3);
      const topSymptoms = symptoms.length ? symptoms : topK.map(([cond]) => cond);
      const [bestCond, bestScore] = sorted.length ? sorted[0] : [disease, 1];

      // Handle confidence score - check if it's already a percentage or decimal
      let confidenceValue = processedData.confidence_score;
      if (confidenceValue !== undefined) {
        // If it's a decimal (0-1), convert to percentage
        confidenceValue = confidenceValue <= 1 ? Math.round(confidenceValue * 100) : Math.round(confidenceValue);
      } else if (bestScore !== undefined) {
        confidenceValue = bestScore <= 1 ? Math.round(bestScore * 100) : Math.round(bestScore);
      } else {
        confidenceValue = 100;
      }

      const specialtyMap = {
        // X-ray conditions
        Atelectasis: 'Pulmonologist',
        Cardiomegaly: 'Cardiologist',
        Effusion: 'Pulmonologist',
        Infiltration: 'Pulmonologist',
        Mass: 'Oncologist',
        Nodule: 'Pulmonologist',
        Pneumonia: 'Pulmonologist',
        Pneumothorax: 'Thoracic Surgeon',
        Consolidation: 'Pulmonologist',
        Edema: 'Cardiologist',
        Emphysema: 'Pulmonologist',
        Fibrosis: 'Pulmonologist',
        Pleural_Thickening: 'Pulmonologist',
        Hernia: 'General Surgeon',
        // CT conditions
        Tumor: 'Oncologist',
        // MRI conditions
        Diabetes: 'Endocrinologist',
        Depression: 'Psychiatrist',
        'Heart Disease': 'Cardiologist',
        'Pleural Effusion': 'Pulmonologist',
        'Glioma': 'Neuro-Oncologist',
        'Meningioma': 'Neurosurgeon',
        'Pituitary Tumor': 'Endocrinologist',
      };

      // Prefer specialist from backend, fallback to map
      const specialty = processedData.specialist || specialtyMap[disease] || specialtyMap[bestCond] || 'General Physician';
      
      console.log('🎯 Specialist selection:');
      console.log('  processedData.specialist:', processedData.specialist);
      console.log('  disease:', disease);
      console.log('  bestCond:', bestCond);
      console.log('  final specialty:', specialty);

      const formattedReport = {
        symptoms: topSymptoms,
        diagnosis: disease || bestCond,
        confidence: confidenceValue,
        findings: processedData.findings || predictionData || [],  // Model predictions
        recommendations: processedData.recommendations || [
          `Consult a ${specialty}`,
          'Follow a healthy lifestyle',
          'Get relevant tests done',
        ],
        suggested_tests: processedData.suggested_tests || [
          'Blood Test',
          'Imaging',
          'Consultation'
        ],
        specialty,
        timestamp: new Date().toISOString(),
        report: reportText,
      };

      setReportData(formattedReport);
      setLoading(false);
      return;
    }

    // Fallback: if no processedData, fetch from API as before
    if (!cleanType) {
      setError('No image type specified and no data passed.');
      setLoading(false);
      return;
    }

    const fetchReport = async () => {
      try {
        const [predictionRes, reportRes] = await Promise.all([
          fetch(`${BASE_API_URL}/predict/${cleanType}/`),
          fetch(`${BASE_API_URL}/generate-report/${cleanType}/`),
        ]);

        if (!predictionRes.ok || !reportRes.ok) {
          throw new Error('One of the API calls failed');
        }

        const predictionData = await predictionRes.json();
        const reportDataRaw = await reportRes.json();

        const sorted = predictionData.predictions.sort((a, b) => b[1] - a[1]);
        const topK = sorted.slice(0, 3);
        const symptoms = topK.map(([cond]) => cond);
        const [bestCond, bestScore] = sorted[0];

        const specialtyMap = {
          Diabetes: 'Endocrinologist',
          Pneumonia: 'Pulmonologist',
          Depression: 'Psychiatrist',
          'Heart Disease': 'Cardiologist',
          'Pleural Effusion': 'Pulmonologist',
        };

        const specialty = specialtyMap[bestCond] || 'General Physician';

        const formattedReport = {
          symptoms,
          diagnosis: reportDataRaw.disease || bestCond,
          confidence: Math.round(bestScore * 100),
          recommendations: [
            `Consult a ${specialty}`,
            'Follow a healthy lifestyle',
            'Get relevant tests done',
          ],
          suggested_tests: ['Blood Test', 'Imaging', 'Consultation'],
          specialty,
          timestamp: new Date().toISOString(),
          report: reportDataRaw.report || '',
        };

        setReportData(formattedReport);
      } catch (err) {
        console.error(err);
        setError('Failed to load report. Please try again.');
      } finally {
        setLoading(false);
      }
    };

    fetchReport();
  }, [cleanType, processedData]);

  if (loading) {
    return (
      <div className="p-6 text-center text-slate-500 min-h-screen">Loading report...</div>
    );
  }

  if (error) {
    return (
      <div className="p-6 text-center text-2xl text-red-600 opacity-60 min-h-screen">
        {error} 😢
      </div>
    );
  }

  return (
    <div className="p-6 min-h-screen space-y-6 max-w-4xl mx-auto">
      <div id="report-content">
        <ReportCard report={reportData} />
      </div>
    </div>
  );
};

export default ResultsPage;
