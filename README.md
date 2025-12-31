# 🩺 MedVision AI - Medical Assistant AI Platform

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-00a393.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61dafb.svg)](https://react.dev)

## 📸 Overview

**MedVision AI** is an intelligent medical diagnostic assistant that analyzes medical images (X-rays, CT scans, MRIs, and ultrasounds), generates comprehensive AI-powered medical reports, and provides follow-up actions including location-based doctor search and AI chatbot assistance.

---

## ✨ Key Features

### 🩻 Medical Image Analysis
* **Multi-Modal Support**: X-ray, CT (2D & 3D), MRI (2D & 3D), and Ultrasound
* **AI-Powered Diagnosis**: Deep learning models for disease detection and classification
* **Detailed Reports**: Gemini AI-generated comprehensive medical reports with findings, diagnoses, and recommendations
* **Image Preview**: Real-time image preview with preprocessing visualization

### 🤖 Dual Chatbot System
* **Public Help Assistant** 🌐: No login required
  - Platform information and features
  - Usage guidance and support
  - Contact information
  - Redirects medical questions to authenticated chatbot
  
* **Medical AI Chatbot** 🩺: User authentication required
  - Medical conditions and disorders information
  - Report interpretation and explanation
  - Symptoms and treatment guidance
  - Powered by Gemini AI

### 👤 User Management
* **Secure Authentication**: JWT-based authentication system
* **User Accounts**: Sign up, login, and profile management
* **Report History**: Personalized history of all medical reports and analyses
* **Session Management**: Secure token-based sessions

### 🗺️ Doctor Search Integration
* **Location-Based Search**: Find doctors and medical facilities nearby
* **Interactive Map**: Leaflet-powered map with doctor locations
* **Specialty Filtering**: Search by medical specialty
* **Contact Information**: Direct access to phone numbers and addresses

### 📄 Export & Download
* **PDF Generation**: Professional medical reports with branding
* **Watermark Support**: Secure document verification
* **Print-Ready Format**: Optimized for printing and archival

---

## 🚀 Tech Stack

### Frontend 🎨
| Technology | Purpose |
|------------|---------|
| **React 18** | UI framework |
| **Vite** | Build tool & dev server |
| **TailwindCSS** | Styling framework |
| **ShadCN-UI** | Component library (Radix-based) |
| **Leaflet** | Interactive maps |
| **@react-pdf/renderer** | PDF generation |
| **React Router** | Client-side routing |
| **Framer Motion** | Animations |

### Backend ⚙️
| Technology | Purpose |
|------------|---------|
| **FastAPI** | Web framework |
| **PyTorch** | Deep learning inference |
| **Gemini Pro AI** | Report generation & chatbot |
| **SQLAlchemy** | ORM & database management |
| **SQLite** | Database |
| **Python 3.10+** | Runtime |

### ML & Image Processing 🧠
| Technology | Purpose |
|------------|---------|
| **timm** | Pretrained vision models |
| **PyDicom** | DICOM file handling |
| **OpenCV** | Image processing |
| **NIbabel** | NIfTI file handling (3D scans) |
| **NumPy** | Numerical computation |
| **Pillow** | Image manipulation |

### Authentication & Security 🔐
| Technology | Purpose |
|------------|---------|
| **JWT** | Token-based authentication |
| **bcrypt** | Password hashing |
| **OAuth2** | Authentication flow |
| **CORS** | Cross-origin resource sharing |

---

## 📂 Project Structure

```
medvision-ai/
├── frontend/                  # React frontend
│   ├── src/
│   │   ├── components/        # Reusable UI components
│   │   │   ├── ChatSupport.jsx
│   │   │   ├── ImageViewer.jsx
│   │   │   └── ...
│   │   ├── context/           # React contexts
│   │   ├── LandingPage.jsx    # Home page
│   │   ├── LoginPage.jsx      # User login
│   │   ├── SignupPage.jsx     # User registration
│   │   ├── UploadPage.jsx     # Image upload
│   │   ├── ResultPage.jsx     # Analysis results
│   │   ├── ResultChatPage.jsx # Medical chatbot
│   │   ├── HistoryPage.jsx    # Report history
│   │   ├── DoctorSearchPage.jsx # Doctor finder
│   │   └── App.jsx            # Main app component
│   ├── package.json
│   └── vite.config.js
│
├── backend/                   # FastAPI backend
│   ├── services/              # Model processing
│   │   ├── xray_service.py
│   │   ├── ct_service.py
│   │   ├── mri_service.py
│   │   └── ultrasound_service.py
│   ├── models/                # ML model files
│   ├── main.py               # Main API routes
│   ├── auth.py               # Authentication logic
│   ├── auth_routes.py        # Auth endpoints
│   ├── database.py           # Database setup
│   ├── db_models.py          # SQLAlchemy models
│   ├── schemas.py            # Pydantic schemas
│   ├── requirements.txt
│   └── medvision.db          # SQLite database
│
├── sample_reports/           # Sample generated reports
├── LICENSE                   # MIT License
└── README.md                 # This file
```

---

## 🔗 API Documentation

### 🔐 Authentication Endpoints

#### `POST /auth/signup`
Register a new user account

**Request Body:**
```json
{
  "email": "user@example.com",
  "full_name": "John Doe",
  "password": "securePassword123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGci...",
  "token_type": "bearer"
}
```

#### `POST /auth/login`
Login with existing credentials

**Request Body:**
```json
{
  "username": "user@example.com",
  "password": "securePassword123"
}
```

#### `GET /auth/me`
Get current user information (requires authentication)

---

### 🩻 X-Ray Endpoints

#### `POST /predict/xray/`
Upload X-ray image for disease prediction

**Headers:** `Authorization: Bearer <token>`

**Form Data:**
- `file`: X-ray image (JPEG, PNG, DICOM)

**Response:**
```json
{
  "predictions": [
    {"disease": "Pneumonia", "confidence": 0.87},
    {"disease": "Normal", "confidence": 0.13}
  ],
  "image_preview": "base64_encoded_string"
}
```

#### `GET /get_latest_results/`
Retrieve most recent X-ray prediction

---

### 🧠 CT Scan Endpoints

#### `POST /predict/ct/2d/`
Upload 2D CT slice for analysis

**Headers:** `Authorization: Bearer <token>`

#### `POST /predict/ct/3d/`
Upload 3D CT volume (NIfTI format)

**Headers:** `Authorization: Bearer <token>`

#### `GET /predict/ct/2d/`, `GET /predict/ct/3d/`
Retrieve latest CT reports

---

### 🧲 MRI Endpoints

#### `POST /predict/mri/2d/`
Upload 2D MRI slice

**Headers:** `Authorization: Bearer <token>`

#### `POST /predict/mri/3d/`
Upload 3D MRI volume (NIfTI format)

**Headers:** `Authorization: Bearer <token>`

#### `GET /predict/mri/3d/`
Retrieve latest MRI report

---

### 🔊 Ultrasound Endpoints

#### `POST /predict/ultrasound/`
Upload ultrasound image for analysis

**Headers:** `Authorization: Bearer <token>`

#### `GET /get_latest_report/ultrasound/`
Retrieve latest ultrasound report

---

### 📋 Report Management

#### `POST /generate-report/{modality}/`
Generate detailed medical report

**Path Parameters:**
- `modality`: One of `xray`, `ct`, `mri`, `ultrasound`

**Headers:** `Authorization: Bearer <token>`

#### `GET /get-latest-report/{modality}/`
Retrieve latest report for specified modality

---

### 💬 Chatbot Endpoints

#### `POST /public_chat/`
Public platform help chatbot (no authentication required)

**Request Body:**
```json
{
  "message": "What features does MedVision AI offer?"
}
```

#### `POST /chat_with_report/`
Medical AI chatbot (requires authentication)

**Headers:** `Authorization: Bearer <token>`

**Request Body:**
```json
{
  "message": "Can you explain my X-ray findings?",
  "report_context": {
    "modality": "xray",
    "findings": "..."
  }
}
```

---

### 📚 History Endpoints

#### `GET /history`
Get all reports for authenticated user

**Headers:** `Authorization: Bearer <token>`

#### `POST /history`
Save a report to user's history

**Headers:** `Authorization: Bearer <token>`

---

## 🛠️ Installation & Setup

### Prerequisites
- **Node.js**: v18+ and npm
- **Python**: 3.10+
- **Git**: For cloning the repository

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/medvision-ai.git
cd medvision-ai/main
```

### 2️⃣ Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create a .env file and add:
# GEMINI_API_KEY=your_gemini_api_key_here

# Run the backend server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at `http://localhost:8000`

### 3️⃣ Frontend Setup

```bash
# Navigate to frontend directory
cd ../frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:5173`

---

## 🧪 Backend Dependencies

```txt
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.12
pydicom>=2.4.0
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
scikit-learn>=1.3.0
pydantic>=2.5.0
python-multipart>=0.0.6
google-generativeai>=0.3.0
nibabel>=5.2.0
geopy>=2.4.0
sqlalchemy>=2.0.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-dotenv>=1.0.0
```

---

## 🎨 Frontend Dependencies

Key packages:
- `react` & `react-dom`: ^18.2.0
- `vite`: ^5.0.0
- `tailwindcss`: ^3.4.0
- `react-router-dom`: ^6.21.0
- `@react-pdf/renderer`: ^3.1.0
- `leaflet` & `react-leaflet`: ^4.2.0
- `@radix-ui/*`: ShadCN component library
- `framer-motion`: ^10.0.0
- `axios`: ^1.6.0

---

## 📸 Screenshots

### Landing Page
<img src="0.png" alt="Landing Page" />

### File Upload Interface
<img src="1.png" alt="Upload Page" />

### Analysis Results
<img src="2.png" alt="Results Page" />

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the `backend` directory:

```env
# Gemini AI API Key (required)
GEMINI_API_KEY=your_api_key_here

# Database (optional - defaults to SQLite)
DATABASE_URL=sqlite:///./medvision.db

# JWT Secret (optional - auto-generated if not provided)
SECRET_KEY=your_secret_key_here
```

### Frontend Configuration

Update API endpoint in `frontend/src` if needed:

```javascript
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
```

---

## 🧑‍💻 Development

### Running Tests

```bash
# Backend tests
cd backend
python -m pytest tests/

# Frontend tests
cd frontend
npm test
```

### Code Formatting

```bash
# Backend (using black)
black backend/

# Frontend (using prettier)
npm run format
```

---

## 🚀 Deployment

### Backend Deployment (Example with Docker)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Deployment

```bash
cd frontend
npm run build
# Deploy the 'dist' folder to your hosting service
```

---

## 💡 Future Enhancements

- [ ] **WebSocket Integration**: Real-time prediction updates
- [ ] **Patient History Management**: Complete medical history tracking
- [ ] **Admin Dashboard**: Analytics, logs, and user management
- [ ] **Appointment Booking**: Google Calendar integration
- [ ] **Multi-language Support**: Reports in multiple languages
- [ ] **Mobile App**: React Native mobile application
- [ ] **DICOM Viewer**: Advanced medical image viewer with annotations
- [ ] **Multi-User Collaboration**: Share and collaborate on reports
- [ ] **Advanced Analytics**: Disease trend analysis and insights
- [ ] **EMR/EHR Integration**: Connect with hospital systems

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

Please ensure:
- Code follows project style guidelines
- All tests pass
- Documentation is updated
- Commit messages are descriptive

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## ✍️ Author

**Mohnish S Shetty**

---

## 🙏 Acknowledgments

- **Google Gemini AI** for powerful language model capabilities
- **PyTorch** team for the deep learning framework
- **FastAPI** community for the excellent web framework
- **React** and **Vite** teams for modern frontend tools
- **ShadCN** for beautiful UI components
- All open-source contributors

---

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/mohnish-srshetty/medvision-ai/issues)


---

<div align="center">

🧱 **Designed and Developed by MedVision** ⚙️

⭐ **Star this repository if you find it helpful!** ⭐

</div>
