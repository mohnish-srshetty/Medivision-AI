import React, { useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import LandingPage from './LandingPage';
import UploadPage from './UploadPage';
import ResultPage from './ResultPage';
import Header from './components/Header';
import Footer from './components/Footer';
import Contact from './Contact';
import ResultChatPage from './ResultChatPage';
import DoctorSearchPage from './DoctorSearchPage';
import LoginPage from './LoginPage';
import SignupPage from './SignupPage';
import HistoryPage from './HistoryPage';
import ProtectedRoute from './components/ProtectedRoute';

function App() {
  // Global state for selected image type and processed data
  const [selectedImageType, setSelectedImageType] = useState(null);
  const [processedData, setProcessedData] = useState(null);

  return (
    <AuthProvider>
      <div className="flex flex-col min-h-screen bg-gray-50 dark:bg-gray-900">
        <Header />
        <main className="flex-grow">
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route
              path="/upload"
              element={
                <ProtectedRoute>
                  <UploadPage
                    selectedImageType={selectedImageType}
                    setSelectedImageType={setSelectedImageType}
                    setProcessedData={setProcessedData}
                  />
                </ProtectedRoute>
              }
            />
            <Route
              path="/results"
              element={
                <ProtectedRoute>
                  <ResultPage
                    processedData={processedData}
                    selectedImageType={selectedImageType}
                  />
                </ProtectedRoute>
              }
            />
            <Route path="/chat" element={
              <ProtectedRoute>
                <ResultChatPage />
              </ProtectedRoute>
            } />
            <Route path="/search-doctor" element={
              <ProtectedRoute>
                <DoctorSearchPage />
              </ProtectedRoute>
            } />
            <Route path='/contact' element={<Contact />} />
            <Route path="/login" element={<LoginPage />} />
            <Route path="/signup" element={<SignupPage />} />
            <Route path="/history" element={
              <ProtectedRoute>
                <HistoryPage />
              </ProtectedRoute>
            } />
          </Routes>
        </main>
        <Footer />
      </div>
    </AuthProvider>
  );
}

export default App;
