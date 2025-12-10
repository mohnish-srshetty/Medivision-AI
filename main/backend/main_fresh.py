from fastapi import FastAPI, UploadFile, File, HTTPException ,Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import shutil
from contextlib import asynccontextmanager
import os
from services.xray_service import process_xray, init_xray_model
from pydantic import BaseModel
from typing import List, Optional
from fastapi import FastAPI, Query
import httpx
from typing import List, Tuple
from dotenv import load_dotenv
import io
import re
import numpy as np
import nibabel as nib # type: ignore
from PIL import Image
import pydicom # type: ignore
from nibabel.loadsave import load # type: ignore
from geopy.geocoders import Nominatim



# Load environment variables
load_dotenv(dotenv_path="../../.env")

# Import your ML model functions for each modality
from services.xray_service import process_xray, init_xray_model
# Uncomment when available:
from services.ct_service import process_ct, init_ct_models
from services.ultrasound_service import process_ultrasound, init_ultrasound_model
from services.mri_service import process_mri, init_mri_models

# Initialize Groq Client
# pip install groq
from groq import Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# Database imports
from database import SessionLocal, engine, Base
import db_models
from auth_routes import register_auth_routes

# Create database tables
db_models.Base.metadata.create_all(bind=engine)

# Global: store latest predictions for frontend polling
latest_xray_results: dict = {}
latest_reports = {}  

# Startup: initialize all models
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_xray_model()
    init_ct_models()
    init_ultrasound_model()
    init_mri_models()
    yield
    print("Shutting down models...")

app = FastAPI(lifespan=lifespan)

# CORS settings
origins = ["*"]  # allow all origins for simplicity; adjust as needed

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # allows requests from these origins
    allow_credentials=True,
    allow_methods=["*"],    # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],    # allow all headers
)


# Register authentication routes
register_auth_routes(app)


PROMPT_TEMPLATES = {
    "xray": (
        """"
        You are a medical AI assistant. 
        Based on the image and the patient symptoms: {symptoms} your task is to:

        1. Idenify if the image is of a chest X-ray. If not, return "Not a chest X-ray" and do not proceed.
        2. Identify the disease with the highest confidence score.
        3. Generate a clear and concise diagnosis statement indicating which disease is most likely present based on the AI analysis.
        4. Mention the confidence score as a percentage.
        5. Include a disclaimer that this is a preliminary AI-based diagnosis and advise the user to consult a healthcare professional for confirmation.
        6. Do not begin with "Based on the image and the patient symptoms" or any other introductory phrase.
        7. Report size should be always between 200 and 300 words.
        8. Use the following format for the output:


        Example Output:
        Disease Expected: Mass
        The AI model analyzed the chest X-ray image and determined that the most likely condition present is Mass, with a confidence score of 47.00%. 
        This suggests there may be an abnormal growth or lump in the lung area that requires further attention. Masses can range from benign 
        (non-cancerous) to malignant (cancerous), so additional medical evaluation such as a CT scan or biopsy may be recommended to determine its 
        nature. This result is an early indication provided by an AI system and should not replace professional medical advice or diagnosis. 
        Please consult a certified radiologist or doctor.

        """
    
    ),
    "ct": (
        '''
        You are a medical AI assistant specialized in interpreting 3D and 2D CT scan results. 
        Given a set of AI-generated confidence scores for tumor detection, your task is to:

        1. Identify whether a tumor or no tumor is more likely based on the highest confidence score.
        2. Clearly mention the detected condition and the confidence score as a percentage (e.g., 92.00%).
        3. Explain what this result means for the patient in clear, simple language.
        4. Describe briefly how 3D CT scans assist in detecting tumors by providing detailed cross-sectional views of the body.
        5. Recommend possible next steps such as further imaging or biopsy for confirmation.
        6. End with a disclaimer stating that this is an AI-generated preliminary result and must be verified by a certified medical professional.
        7. Do not begin with "Based on the image and the patient symptoms" or any other introductory phrase.
        8. Report size should be always between 200 and 300 words.
        9. Use the following format for the output:

        """
        Output example 
        Condition Detected: Tumor
        The AI analysis of your 3D CT scan of the brain indicates a high probability of a tumor, with a confidence score of 92.00%. This suggests there may be an abnormal mass or growth
        present in the scanned region. 3D CT scans allow doctors to view detailed cross-sectional images of internal tissues, making it easier to identify potential issues like 
        tumors. While this result is a strong indicator, it is not a confirmed diagnosis. Further testing, such as an MRI or biopsy, may be required. 
        Disclaimer: This is an AI-generated summary. Please consult a certified doctor or radiologist for medical confirmation and advice.
        '''
        "Based on the image and the patient symptoms: {symptoms}, "
        "produce a detailed CT scan report including observations, differential diagnoses, and next steps."
    ),
    "ultrasound": (
        '''

        You are a medical assistant specialized in interpreting ultrasound scan results. 
        Based on the image and the patient symptoms: {symptoms}, your task is to:

        1. Identify the most likely condition based on the highest confidence score from the following categories: Normal, Cyst, Mass, Fluid, Other Anomaly.
        2. Clearly state the detected condition along with its confidence score as a percentage (e.g., 88.50%).
        3. Explain in simple and compassionate language what the result implies for the patient.
        4. Provide a brief explanation of how ultrasound helps in detecting such conditions using sound waves for real-time internal imaging.
        5. Suggest next medical steps such as follow-up scans, consultations, or further diagnostic procedures.
        6. End with a disclaimer stating that this is an AI-generated preliminary result and must be verified by a certified medical professional.
        7. Do not begin with "Based on the image and the patient symptoms" or any other introductory phrase.
        8. Report size should be always between 200 and 300 words.
        9. Use the following format for the output:
        
        Example Output:
        Condition Detected: Cyst

        Based on the ultrasound image, the AI model has identified the most likely condition as a Cyst, with a confidence score of 92.30%. This suggests the presence of a fluid-filled sac, which is typically benign and may not cause symptoms. Ultrasound imaging uses sound waves to create real-time visuals of internal organs and is effective in detecting such abnormalities. While most cysts are harmless, a follow-up consultation with a healthcare professional is recommended to evaluate its size, nature, and whether further investigation is needed.

        Disclaimer: This is an AI-generated summary and not a substitute for professional medical advice.

        Output the result as one clear and concise paragraph of around 100 words for easy understanding by non-medical users.
        '''
        
        "generate a comprehensive ultrasound report covering findings, clinical significance, and recommendations."
    ),
    "mri": (
        "You are a radiology report assistant specialized in interpreting MRI scans. "
        "Based on the image and the patient symptoms: {symptoms}, "
        "create a detailed MRI report including key findings, interpretation, and suggested follow‑up."
    ),
}
# A generic fallback if you ever get an unexpected modality:
FALLBACK_TEMPLATE = (
    "You are a medical report assistant. Based on the image and patient symptoms: {symptoms}, "
    "generate a concise professional report including findings and recommendations."
)

# Utility: parse report sections for recommendations and suggested tests
def parse_report_sections(report: str) -> dict:
    """
    Parses the report to extract findings, recommendations and suggested tests.
    Returns a dict with 'findings', 'recommendations', and 'suggested_tests' as lists.
    """
    sections = {
        "findings": [],
        "recommendations": [],
        "suggested_tests": []
    }
    
    # Extract findings section - can be multiple lines or bullet points
    findings_match = re.search(r"FINDINGS:\s*([^\n]+(?:\n(?!RECOMMENDATIONS:|SUGGESTED TESTS:|DISCLAIMER:)[^\n]+)*)", report, re.IGNORECASE)
    if findings_match:
        findings_text = findings_match.group(1).strip()
        # Check if it's a numbered or bulleted list
        if re.search(r"^\d+\.", findings_text, re.MULTILINE) or re.search(r"^[-•]\s", findings_text, re.MULTILINE):
            # Parse as list
            sections["findings"] = [
                re.sub(r"^[\d+\.\-•]\s*", "", line.strip())
                for line in findings_text.split("\n")
                if line.strip() and (re.match(r"^\d+\.", line.strip()) or re.match(r"^[-•]\s", line.strip()))
            ]
        else:
            # Single finding or paragraph - split by sentences or keep as single item
            sections["findings"] = [findings_text] if findings_text else []
    
    # Extract recommendations
    recs_match = re.search(r"RECOMMENDATIONS:\s*((?:\d+\.\s*[^\n]+\n?)+)", report, re.IGNORECASE)
    if recs_match:
        recs_text = recs_match.group(1)
        sections["recommendations"] = [
            re.sub(r"^\d+\.\s*", "", line.strip()) 
            for line in recs_text.split("\n") 
            if line.strip() and re.match(r"^\d+\.", line.strip())
        ]
    
    # Extract suggested tests
    tests_match = re.search(r"SUGGESTED TESTS:\s*((?:\d+\.\s*[^\n]+\n?)+)", report, re.IGNORECASE)
    if tests_match:
        tests_text = tests_match.group(1)
        sections["suggested_tests"] = [
            re.sub(r"^\d+\.\s*", "", line.strip()) 
            for line in tests_text.split("\n") 
            if line.strip() and re.match(r"^\d+\.", line.strip())
        ]
    
    return sections

# Utility: extract top-k symptom labels
def extract_top_symptoms(predictions: List[Tuple[str, float]], top_k: int = 3) -> List[str]:
    sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
    return [label for label, _ in sorted_preds[:top_k]]

# Generate report using Groq with Llama
def generate_medical_report(symptoms: List[str], image_bytes: bytes, modality: str, mime_type: Optional[str] = None) -> str:
    # Prepare prompt
    template = PROMPT_TEMPLATES.get(modality.lower(), FALLBACK_TEMPLATE)
    prompt = template.format(symptoms=", ".join(symptoms))
    
    # Note: Groq's Llama models don't support image inputs in the same way as Gemini
    # We'll use text-only prompts based on the model predictions
    # If vision is needed, use llama-3.2-90b-vision-preview
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        
        if not response or not response.choices or not response.choices[0].message.content:
            raise HTTPException(status_code=500, detail="Empty response from Groq API.")
        
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")


@app.post("/predict/xray/")
async def predict_xray(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/bmp"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Get predictions from model
        predictions = process_xray(temp_path, device="cpu")
        os.remove(temp_path)
        
        print(f"🔍 X-ray - Predictions: {predictions}")
        
        global latest_xray_results
        latest_xray_results = {label: float(prob) for label, prob in predictions}
        
        # Get top prediction for Groq
        label, prob = predictions[0] if predictions else ("Unknown", 0.0)
        
        # Build list of predictions for context
        all_predictions_str = "\n".join([f"- {lbl}: {p*100:.2f}%" for lbl, p in predictions[:5]])
       
        prompt = f"""
You are a medical specialist in interpreting chest X-ray results.

The AI model has analyzed the chest X-ray and provided the following predictions:
{all_predictions_str}

Top prediction: {label} with {prob*100:.2f}% confidence.

Generate a structured medical report with the following sections:

1. **SPECIALIST**: Based on the diagnosis of {label}, recommend the MOST APPROPRIATE medical specialist the patient should consult. Provide ONLY the specialist name (e.g., "Pulmonologist", "Cardiologist", "General Physician", "Radiologist").

2. **Explanation**: Provide a clear, compassionate explanation of what {label} means for the patient in 2-3 sentences. Explain how X-rays help detect such conditions.

3. **RECOMMENDATIONS**: Provide EXACTLY 3 disorder-specific clinical recommendations based on {label}. These should be:
   - Specific to {label} management and treatment
   - Actionable next steps for the patient
   - Include specialist consultation if needed

4. **SUGGESTED TESTS**: Suggest EXACTLY 3 diagnostic tests or procedures specific to {label}:
   - Confirmatory tests for {label}
   - Additional imaging if relevant
   - Lab tests or other procedures specific to this diagnosis

5. **DISCLAIMER**: Include a standard medical AI disclaimer

Format your response EXACTLY as follows:

Condition Detected: {label}

SPECIALIST: [Specialist name only]

[2-3 sentence explanation paragraph about what this means and how X-ray helps]

RECOMMENDATIONS:
1. [Specific recommendation for {label}]
2. [Specific recommendation for {label}]
3. [Specific recommendation for {label}]

SUGGESTED TESTS:
1. [Specific test for {label}]
2. [Specific test for {label}]
3. [Specific test for {label}]

DISCLAIMER: This is an AI-generated preliminary assessment. Please consult a qualified radiologist or physician for professional medical advice and confirmation.
"""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024
        )
        report = response.choices[0].message.content or "<empty>"
        
        # Extract disease from report
        match = re.search(r"Condition Detected:\s*(.+)", report)
        disease = match.group(1).strip() if match else label
        
        # Extract specialist from report
        specialist_match = re.search(r"SPECIALIST:\s*(.+)", report)
        specialist = specialist_match.group(1).strip() if specialist_match else "Pulmonologist"
        
        # Parse structured sections
        sections = parse_report_sections(report)

        # Build response with all necessary fields
        latest_reports["xray"] = {
            "Symptom": label,
            "disease": disease,
            "confidence_score": float(prob),  # FROM MODEL
            "report": report,
            "specialist": specialist,  # FROM GROQ
            "findings": [[lbl, float(p)] for lbl, p in predictions],  # FROM MODEL
            "recommendations": sections["recommendations"],
            "suggested_tests": sections["suggested_tests"],
            "predictions": [[lbl, float(p)] for lbl, p in predictions]
        }
        
        return JSONResponse(content=latest_reports["xray"])
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preview/")
async def preview_dicom(file: UploadFile = File(...)):
    """Preview DICOM files by converting to PNG"""
    temp_path = f"temp_preview_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        ds = pydicom.dcmread(temp_path)
        try:
            ds.decompress()
        except:
            pass
        
        pixel_array = ds.pixel_array
        img_normalized = ((pixel_array - pixel_array.min()) / 
                         (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        
        pil_img = Image.fromarray(img_normalized)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        img_bytes = io.BytesIO()
        pil_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        os.remove(temp_path)
        return StreamingResponse(img_bytes, media_type="image/png")
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Preview error: {str(e)}")


@app.get("/get_latest_results/")
async def get_latest_results():
    if not latest_xray_results:
        return {"message": "No prediction results available yet."}
    return latest_xray_results


@app.post("/generate-report/{modality}/")
async def generate_report(
    modality: str = Path(..., description="One of: xray, ct, ultrasound, mri"),
    file: UploadFile = File(...)
):
    modality = modality.lower()
    if modality not in ["xray", "ct", "ultrasound", "mri"]:
        raise HTTPException(status_code=400, detail="Invalid modality.")
    if file.content_type not in ["image/jpeg", "image/png", "image/bmp"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    temp_path = f"temp_{modality}_{file.filename}"
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    try:
        # Inference dispatch
        if modality == "xray":
            raw_preds = process_xray(temp_path, device="cpu")
        # elif modality == "ct": raw_preds = process_ct(temp_path, device="cpu")
        # elif modality == "ultrasound": raw_preds = process_ultrasound(temp_path, device="cpu")
        # else: raw_preds = process_mri(temp_path, device="cpu")

        symptoms = extract_top_symptoms(raw_preds)
        # Read bytes
        with open(temp_path, "rb") as f:
            img_bytes = f.read()
        os.remove(temp_path)

        report = generate_medical_report(symptoms, img_bytes, modality)
        # Extract the disease from the report
        match = re.search(r"Disease Expected:\s*(.+)", report)
        disease = match.group(1).strip() if match else "Unknown"
        # Store the report in a global variable
        latest_reports[modality] = {
        "disease": disease,
        "symptoms": symptoms,
        "report": report
        
        }
        return JSONResponse(content={"symptoms": symptoms, "disease": disease ,"report": report})
    except HTTPException:
        os.remove(temp_path)
        raise
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/get-latest-report/{modality}/")
async def get_latest_report(modality: str = Path(...)):
    modality = modality.lower()
    if modality not in latest_reports:
        raise HTTPException(status_code=404, detail="No report available for this modality.")
    return latest_reports[modality]


# CT 2D and 3D routes
@app.post("/predict/ct/2d/")
async def generate_report_ct2d(file: UploadFile = File(...)):
    modality = "ct"
    mode = "2d"

    # Only allow image files for 2D slices
    if file.content_type not in ["image/jpeg", "image/png", "image/bmp"]:
        raise HTTPException(status_code=400, detail="Unsupported file type for CT2D.")

    temp_path = f"temp_ct2d_{file.filename}"
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        # Inference
        raw_preds = process_ct(temp_path, mode=mode, device="cpu")
        symptoms = extract_top_symptoms(raw_preds)

        # Read image bytes before deleting temp
        with open(temp_path, "rb") as f:
            img_bytes = f.read()
        os.remove(temp_path)

        # Generate report using correct MIME type
        report = generate_medical_report(
            symptoms, img_bytes, modality=modality, mime_type=file.content_type
        )

        # Extract disease
        match = re.search(r"Condition Detected:\s*(.+)", report)
        disease = match.group(1).strip() if match else "Unknown"

        # Store
        latest_reports["ct2d"] = {
            "symptoms": symptoms,
            "disease": disease,
            "report": report
        }

        return JSONResponse({
    if temp_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
        raise HTTPException(status_code=400, detail="MRI 3D requires volumetric files (.dcm, .nii, .nii.gz)")
    
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    
    try:
        # Get all predictions from model (not just top-1)
        raw_preds = process_mri(temp_path, mode='3d', device="cpu", top_k=4)  # Get all 4 classes
        label, prob = raw_preds[0]  # Top prediction
        print(f"🔍 MRI 3D - All Predictions: {raw_preds}")
        print(f"🔍 MRI 3D - Top Prediction: {label}, Confidence: {prob*100:.2f}%")
        
        os.remove(temp_path)

        # Generate report with TEXT-ONLY prompt (no images to Groq)
        confidence_str = f"{prob*100:.2f}%"
        
        # Build list of all predictions for context
        all_predictions_str = "\n".join([f"- {lbl}: {p*100:.2f}%" for lbl, p in raw_preds])
        
        # Determine if we should show uncertainty
        show_uncertainty = prob < 0.85
        
        prompt = f"""
You are a medical specialist in interpreting brain MRI scan results.

The AI model has analyzed the brain MRI and provided the following predictions:
{all_predictions_str}

Top prediction: {label} with {confidence_str} confidence.

Generate a structured medical report with the following sections:

1. **SPECIALIST**: Based on the diagnosis of {label}, recommend the MOST APPROPRIATE medical specialist the patient should consult. Provide ONLY the specialist name (e.g., "Neuro-Oncologist", "Neurosurgeon", "Neurologist", "Endocrinologist").

2. **Explanation**: Provide a clear, compassionate explanation of what {label} means for the patient in 2-3 sentences. Explain how brain MRI scans help detect such conditions.

3. **RECOMMENDATIONS**: Provide EXACTLY 3 disorder-specific clinical recommendations based on {label}. These should be:
   - Specific to {label} management and treatment
   - Actionable next steps for the patient
   - Include specialist consultation if needed

4. **SUGGESTED TESTS**: Suggest EXACTLY 3 diagnostic tests or procedures specific to {label}:
   - Confirmatory tests for {label}
   - Additional imaging if relevant
   - Biopsy or other procedures specific to this diagnosis

5. **DISCLAIMER**: Include a standard medical AI disclaimer

Format your response EXACTLY as follows:

Condition Detected: {label}

SPECIALIST: [Specialist name only]

[2-3 sentence explanation paragraph about what this means and how MRI helps]

RECOMMENDATIONS:
1. [Specific recommendation for {label}]
2. [Specific recommendation for {label}]
3. [Specific recommendation for {label}]

SUGGESTED TESTS:
1. [Specific test for {label}]
2. [Specific test for {label}]
3. [Specific test for {label}]

DISCLAIMER: This is an AI-generated preliminary assessment. Please consult a qualified neurologist or radiologist for professional medical advice and confirmation.
"""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024
        )
        report = response.choices[0].message.content or "<empty>"
        
        # Extract disease from report
        match = re.search(r"Condition Detected:\s*(.+)", report)
        disease = match.group(1).strip() if match else label
        
        # Extract specialist from report
        specialist_match = re.search(r"SPECIALIST:\s*(.+)", report)
        specialist = specialist_match.group(1).strip() if specialist_match else "Neurologist"
        
        # Parse structured sections (recommendations and tests only, not findings)
        sections = parse_report_sections(report)

        # Build response with all necessary fields
        latest_reports["mri3d"] = {
            "Symptom": label,
            "disease": disease,
            "confidence_score": float(prob),  # FROM MODEL - Decimal 0-1
            "report": report,
            "specialist": specialist,  # FROM GROQ
            "findings": [[lbl, float(p)] for lbl, p in raw_preds],  # FROM MODEL - All predictions
            "recommendations": sections["recommendations"],  # Array of recommendations
            "suggested_tests": sections["suggested_tests"],  # Array of tests
            "predictions": [[lbl, float(p)] for lbl, p in raw_preds]  # All predictions
        }
        return JSONResponse(latest_reports["mri3d"])
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


## Ultrasound Endpoint
@app.post("/predict/ultrasound/")
async def generate_report_ultrasound(file: UploadFile = File(...)):
    temp_path = f"temp_ultrasound_{file.filename}"
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    
    try:
        # Get all predictions from model
        raw_preds = process_ultrasound(temp_path, device="cpu", top_k=5)
        label, prob = raw_preds[0]
        print(f"🔍 Ultrasound - All Predictions: {raw_preds}")
        print(f"🔍 Ultrasound - Top Prediction: {label}, Confidence: {prob*100:.2f}%")
        
        os.remove(temp_path)
        
        # Build list of all predictions for context
        all_predictions_str = "\n".join([f"- {lbl}: {p*100:.2f}%" for lbl, p in raw_preds])
        
        prompt = f"""
You are a medical specialist in interpreting ultrasound scan results.

The AI model has analyzed the ultrasound and provided the following predictions:
{all_predictions_str}

Top prediction: {label} with {prob*100:.2f}% confidence.

Generate a structured medical report with the following sections:

1. **SPECIALIST**: Based on the diagnosis of {label}, recommend the MOST APPROPRIATE medical specialist the patient should consult. Provide ONLY the specialist name (e.g., "Radiologist", "Gynecologist", "Urologist", "General Surgeon").

2. **Explanation**: Provide a clear, compassionate explanation of what {label} means for the patient in 2-3 sentences. Explain how ultrasound scans help detect such conditions.

3. **RECOMMENDATIONS**: Provide EXACTLY 3 disorder-specific clinical recommendations based on {label}. These should be:
   - Specific to {label} management and treatment
   - Actionable next steps for the patient
   - Include specialist consultation if needed

4. **SUGGESTED TESTS**: Suggest EXACTLY 3 diagnostic tests or procedures specific to {label}:
   - Confirmatory tests for {label}
   - Additional imaging if relevant
   - Biopsy or other procedures specific to this diagnosis

5. **DISCLAIMER**: Include a standard medical AI disclaimer

Format your response EXACTLY as follows:

Condition Detected: {label}

SPECIALIST: [Specialist name only]

[2-3 sentence explanation paragraph about what this means and how ultrasound helps]

RECOMMENDATIONS:
1. [Specific recommendation for {label}]
2. [Specific recommendation for {label}]
3. [Specific recommendation for {label}]

SUGGESTED TESTS:
1. [Specific test for {label}]
2. [Specific test for {label}]
3. [Specific test for {label}]

DISCLAIMER: This is an AI-generated preliminary assessment. Please consult a qualified radiologist or physician for professional medical advice and confirmation.
"""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024
        )
        report = response.choices[0].message.content or "<empty>"
        
        # Extract disease from report
        match = re.search(r"Condition Detected:\s*(.+)", report)
        disease = match.group(1).strip() if match else label
        
        # Extract specialist from report
        specialist_match = re.search(r"SPECIALIST:\s*(.+)", report)
        specialist = specialist_match.group(1).strip() if specialist_match else "Radiologist"
        
        # Parse structured sections
        sections = parse_report_sections(report)

        # Build response with all necessary fields
        latest_reports["ultrasound"] = {
            "Symptom": label,
            "disease": disease,
            "confidence_score": float(prob),  # FROM MODEL
            "report": report,
            "specialist": specialist,  # FROM GROQ
            "findings": [[lbl, float(p)] for lbl, p in raw_preds],  # FROM MODEL
            "recommendations": sections["recommendations"],
            "suggested_tests": sections["suggested_tests"],
            "predictions": [[lbl, float(p)] for lbl, p in raw_preds]
        }
        return JSONResponse(latest_reports["ultrasound"])
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


## CT 2D Endpoint
@app.post("/predict/ct/2d/")
async def generate_report_ct2d(file: UploadFile = File(...)):
    temp_path = f"temp_ct2d_{file.filename}"
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    
    try:
        # Get predictions from model
        raw_preds = process_ct(temp_path, mode="2d", device="cpu")
        # raw_preds is list of tuples: [("No Tumor", 0.45), ("Tumor", 0.55)]
        label, prob = raw_preds[0]
        print(f"🔍 CT 2D - Predictions: {raw_preds}")
        print(f"🔍 CT 2D - Top Prediction: {label}, Confidence: {prob*100:.2f}%")
        
        os.remove(temp_path)
        
        all_predictions_str = "\n".join([f"- {lbl}: {p*100:.2f}%" for lbl, p in raw_preds])
        
        prompt = f"""
You are a medical specialist in interpreting 2D CT scan results.

The AI model has analyzed the CT scan and provided the following predictions:
{all_predictions_str}

Top prediction: {label} with {prob*100:.2f}% confidence.

Generate a structured medical report with the following sections:

1. **SPECIALIST**: Based on the diagnosis of {label}, recommend the MOST APPROPRIATE medical specialist the patient should consult. Provide ONLY the specialist name (e.g., "Oncologist", "Radiologist", "General Physician").

2. **Explanation**: Provide a clear, compassionate explanation of what {label} means for the patient in 2-3 sentences. Explain how CT scans help detect such conditions.

3. **RECOMMENDATIONS**: Provide EXACTLY 3 disorder-specific clinical recommendations based on {label}. These should be:
   - Specific to {label} management and treatment
   - Actionable next steps for the patient
   - Include specialist consultation if needed

4. **SUGGESTED TESTS**: Suggest EXACTLY 3 diagnostic tests or procedures specific to {label}:
   - Confirmatory tests for {label}
   - Additional imaging if relevant
   - Biopsy or other procedures specific to this diagnosis

5. **DISCLAIMER**: Include a standard medical AI disclaimer

Format your response EXACTLY as follows:

Condition Detected: {label}

SPECIALIST: [Specialist name only]

[2-3 sentence explanation paragraph about what this means and how CT helps]

RECOMMENDATIONS:
1. [Specific recommendation for {label}]
2. [Specific recommendation for {label}]
3. [Specific recommendation for {label}]

SUGGESTED TESTS:
1. [Specific test for {label}]
2. [Specific test for {label}]
3. [Specific test for {label}]

DISCLAIMER: This is an AI-generated preliminary assessment. Please consult a qualified radiologist or physician for professional medical advice and confirmation.
"""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024
        )
        report = response.choices[0].message.content or "<empty>"
        
        # Extract disease from report
        match = re.search(r"Condition Detected:\s*(.+)", report)
        disease = match.group(1).strip() if match else label
        
        # Extract specialist from report
        specialist_match = re.search(r"SPECIALIST:\s*(.+)", report)
        specialist = specialist_match.group(1).strip() if specialist_match else "Radiologist"
        
        # Parse structured sections
        sections = parse_report_sections(report)

        # Build response with all necessary fields
        latest_reports["ct2d"] = {
            "Symptom": label,
            "disease": disease,
            "confidence_score": float(prob),  # FROM MODEL
            "report": report,
            "specialist": specialist,  # FROM GROQ
            "findings": [[lbl, float(p)] for lbl, p in raw_preds],  # FROM MODEL
            "recommendations": sections["recommendations"],
            "suggested_tests": sections["suggested_tests"],
            "predictions": [[lbl, float(p)] for lbl, p in raw_preds]
        }
        return JSONResponse(latest_reports["ct2d"])
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))
