import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { useAuth } from './context/AuthContext';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Button } from './components/ui/button';
import { Alert, AlertDescription } from './components/ui/alert';
import { AlertCircle, Upload, FileText, X } from 'lucide-react';
import ImageTypeSelector from './components/ImageTypeSelector';
import DicomViewer from './components/DicomViewer';
import ImageViewer from './components/ImageViewer';

const UploadPage = ({ selectedImageType, setSelectedImageType, setProcessedData }) => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();
  const { token, user } = useAuth();


  const BASE_API_URL = '';

  const handleUpload = async () => {
    if (!file) return setError('Please select a file first.');
    if (!selectedImageType) return setError('Please select an image type first.');

    let predictionEndpoint = '';
    // let reportEndpoint = ''; // Removed as per refactor

    try {
      if (selectedImageType === 'xray') {
        predictionEndpoint = `${BASE_API_URL}/predict/xray/`;
      } else if (selectedImageType === 'ct_2d') {
        predictionEndpoint = `${BASE_API_URL}/predict/ct/2d/`;
      } else if (selectedImageType === 'ct_3d') {
        predictionEndpoint = `${BASE_API_URL}/predict/ct/3d/`;
      } else if (selectedImageType === 'mri_2d') {
        predictionEndpoint = `${BASE_API_URL}/predict/mri/2d/`;
      } else if (selectedImageType === 'mri_3d') {
        predictionEndpoint = `${BASE_API_URL}/predict/mri/3d/`;
      } else if (selectedImageType === 'ultrasound') {
        predictionEndpoint = `${BASE_API_URL}/predict/ultrasound/`;
      } else {
        return setError('Unsupported image type selected.');
      }
    } catch (err) {
      return setError('Invalid image type format.');
    }

    try {
      setUploading(true);
      setError(null);
      setUploadProgress(0);

      const formData = new FormData();
      formData.append('file', file);

      const predictionRes = await axios.post(predictionEndpoint, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Authorization': `Bearer ${token}`
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          setUploadProgress(percentCompleted);
        },
      });

      // All endpoints now return the full report data structure
      const reportData = predictionRes.data;

      setProcessedData({
        predictions: predictionRes.data.predictions || null,
        report: reportData.report,
        disease: reportData.disease,
        symptoms: reportData.symptoms || [],
        recommendations: reportData.recommendations || [],
        suggested_tests: reportData.suggested_tests || [],
        imagePreview: preview,
        imageType: selectedImageType
      });

      // Save to history if logged in
      if (token && user) {
        try {
          // Use the backend's confidence_score for consistency with result page
          let confidence = predictionRes.data.confidence_score || 0;

          const historyPayload = {
            modality: selectedImageType,
            disease_detected: reportData.disease || "Unknown",
            confidence_score: confidence,
            report_text: reportData.report || "No report generated",
            image_path: "placeholder.jpg" // We might want to handle image upload to cloud later
          };

          await axios.post('/history', historyPayload, {
            headers: {
              Authorization: `Bearer ${token}`,
              'Content-Type': 'application/json'
            }
          });
          console.log("Report saved to history");
        } catch (saveError) {
          console.error("Failed to save history:", saveError);
          // Don't block navigation on save error
        }
      }

      navigate('/results', {
        state: {
          selectedImageType,
          processedData: {
            predictions: predictionRes.data.predictions || null,
            findings: predictionRes.data.findings || null,
            confidence_score: predictionRes.data.confidence_score || 0,
            specialist: predictionRes.data.specialist || null,
            report: reportData.report,
            disease: reportData.disease,
            symptoms: reportData.symptoms || [],
            recommendations: reportData.recommendations || [],
            suggested_tests: reportData.suggested_tests || [],
            imagePreview: preview,
          },
        },
      });
    } catch (err) {
      console.error(err);
      const errorMessage = err.response?.data?.detail || 'An error occurred during upload or analysis. Please try again.';
      setError(errorMessage);
    } finally {
      setUploading(false);
    }
  };

  const handleFileChange = async (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      const fileName = selectedFile.name.toLowerCase();
      const isValidFile = selectedFile.type.startsWith('image/') ||
        fileName.endsWith('.dcm') ||
        fileName.endsWith('.tiff') ||
        fileName.endsWith('.tif') ||
        fileName.endsWith('.nii') ||
        fileName.endsWith('.nii.gz');

      if (!isValidFile) {
        setError('Please select a valid image file (JPG, PNG, DICOM, TIFF, NIfTI).');
        return;
      }

      setFile(selectedFile);
      setError(null);

      // For DICOM and NIfTI files, fetch preview from backend
      if (fileName.endsWith('.dcm') || fileName.endsWith('.nii') || fileName.endsWith('.nii.gz')) {
        try {
          const formData = new FormData();
          formData.append('file', selectedFile);

          const response = await fetch('/preview/', {
            method: 'POST',
            body: formData
          });

          if (response.ok) {
            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);
            setPreview(imageUrl);
          } else {
            const errorData = await response.json();
            console.error('Preview error:', errorData);
            setError(`Preview failed: ${errorData.error || 'Unknown error'}`);
            setPreview('dicom_preview_placeholder'); // Fallback to placeholder
          }
        } catch (error) {
          console.error('Preview error:', error);
          setError('Failed to generate preview');
          setPreview('dicom_preview_placeholder'); // Fallback to placeholder
        }
      } else {
        // For regular images, use FileReader
        const reader = new FileReader();
        reader.onloadend = () => {
          setPreview(reader.result);
        };
        reader.readAsDataURL(selectedFile);
      }
    }
  };


  return (
    <Card className="w-full shadow-md min-h-screen">
      <CardHeader>
        <CardTitle className="text-xl font-semibold">Upload Medical Image</CardTitle>
      </CardHeader>

      <CardContent>
        <div className="mb-6">
          <ImageTypeSelector
            selectedImageType={selectedImageType}
            setSelectedImageType={setSelectedImageType}
          />
        </div>

        {error && (
          <Alert variant="destructive" className="mb-4">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <input
          type="file"
          onChange={handleFileChange}
          accept="image/*,.dcm,.tiff,.tif,.nii,.nii.gz"
          className="hidden"
          ref={fileInputRef}
        />

        {preview ? (
          <div className="border-2 border-blue-400 bg-blue-50 rounded-lg p-6 mb-4">
            <div className="relative w-full h-full flex items-center justify-center bg-black rounded-lg overflow-hidden min-h-[400px]">
              {file && file.name.toLowerCase().endsWith('.dcm') ? (
                <DicomViewer file={file} />
              ) : (
                <ImageViewer src={preview} alt="Preview" />
              )}
              <Button
                variant="destructive"
                size="icon"
                className="absolute top-2 right-2 h-8 w-8 rounded-full shadow-md z-10"
                onClick={(e) => {
                  e.stopPropagation();
                  setFile(null);
                  setPreview(null);
                  setError(null);
                  if (fileInputRef.current) fileInputRef.current.value = '';
                }}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>
        ) : (
          <div
            className="border-2 border-dashed border-slate-300 hover:border-blue-400 rounded-lg p-6 mb-4 transition-colors duration-300 cursor-pointer"
            onDragOver={(e) => {
              e.preventDefault();
              e.stopPropagation();
            }}
            onDrop={(e) => {
              e.preventDefault();
              e.stopPropagation();

              const droppedFile = e.dataTransfer.files[0];
              if (!droppedFile?.type.startsWith('image/')) {
                setError('Please upload an image file.');
                return;
              }

              setFile(droppedFile);
              setError(null);

              const reader = new FileReader();
              reader.onloadend = () => {
                setPreview(reader.result);
              };
              reader.readAsDataURL(droppedFile);
            }}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="flex flex-col items-center justify-center h-64 text-slate-400">
              <Upload className="h-12 w-12 mb-2" />
              <p className="text-sm">Drag & drop or click to upload</p>
              <p className="text-xs mt-1">Supports JPG, PNG, DICOM, TIFF, NIfTI (.nii, .nii.gz)</p>
            </div>
          </div>
        )}

        <Button
          className="w-full bg-blue-600 hover:bg-blue-700"
          onClick={handleUpload}
          disabled={!file || !selectedImageType || uploading}
        >
          {uploading ? `Analyzing... ${uploadProgress}%` : 'Analyze Image'}
        </Button>
      </CardContent>
    </Card>
  );
};

export default UploadPage;
