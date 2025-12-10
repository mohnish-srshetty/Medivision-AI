from fastapi import FastAPI, UploadFile, File, HTTPException ,Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import shutil
from contextlib import asynccontextmanager
import os
from pathlib import Path as PathLib
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
    # Text-only prompt is used
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024
    )
    if not response or not response.choices or not response.choices[0].message.content:
        raise HTTPException(status_code=500, detail="Empty response from Groq API.")
    return response.choices[0].message.content


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

IMPORTANT: For the SPECIALIST field, you MUST recommend a specific medical specialist based on the condition {label}. DO NOT suggest "General Physician", "Radiologist", or "General Surgeon" unless absolutely necessary. Provide specific specialists like Pulmonologist, Cardiologist, Thoracic Surgeon, Oncologist, Infectious Disease Specialist, etc.

1. **SPECIALIST**: Based ONLY on {label}, recommend the MOST APPROPRIATE medical specialist. Provide ONLY the specialist name without explanation or extra text.

2. **Explanation**: Provide a clear, compassionate explanation of what {label} means for the patient in 2-3 sentences.

3. **RECOMMENDATIONS**: Provide EXACTLY 3 disorder-specific clinical recommendations based on {label}.

4. **SUGGESTED TESTS**: Suggest EXACTLY 3 diagnostic tests or procedures specific to {label}.

5. **DISCLAIMER**: Include a standard medical AI disclaimer.

Format your response EXACTLY as follows:

Condition Detected: {label}

SPECIALIST: [ONLY the specialist name - e.g., Pulmonologist, Cardiologist, Thoracic Surgeon - NO other text]

[2-3 sentence explanation paragraph]

RECOMMENDATIONS:
1. [Recommendation 1]
2. [Recommendation 2]
3. [Recommendation 3]

SUGGESTED TESTS:
1. [Test 1]
2. [Test 2]
3. [Test 3]

DISCLAIMER: This is an AI-generated preliminary assessment. Please consult a qualified radiologist or physician for professional medical advice.
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
        
        # Extract specialist from report - more robust extraction
        specialist_match = re.search(r"SPECIALIST:\s*(.+?)(?:\n|$)", report, re.IGNORECASE)
        
        # Define condition-to-specialist mapping - this is the PRIMARY source
        condition_specialist_map = {
            "Atelectasis": "Pulmonologist",
            "Cardiomegaly": "Cardiologist",
            "Effusion": "Pulmonologist",
            "Infiltration": "Pulmonologist",
            "Mass": "Oncologist",
            "Nodule": "Pulmonologist",
            "Pneumonia": "Infectious Disease Specialist",
            "Pneumothorax": "Thoracic Surgeon",
            "Consolidation": "Pulmonologist",
            "Edema": "Cardiologist",
            "Emphysema": "Pulmonologist",
            "Fibrosis": "Pulmonologist",
            "Pleural_Thickening": "Pulmonologist",
            "Hernia": "General Surgeon"
        }
        
        # ALWAYS use condition-based specialist - don't rely on Groq
        specialist = condition_specialist_map.get(label, "Pulmonologist")
        print(f"🎯 X-Ray Specialist Mapping: label='{label}' → specialist='{specialist}'")
        print(f"📋 All available conditions in map: {list(condition_specialist_map.keys())}")
        
        # We can still try to get Groq's suggestion for logging, but won't use it
        if specialist_match:
            groq_specialist = specialist_match.group(1).strip().replace("**", "").replace("*", "").strip()
            print(f"📝 Groq suggested: {groq_specialist} (IGNORED - using condition map instead)")
        
        # Parse structured sections
        sections = parse_report_sections(report)

        # Build response with all necessary fields
        response_data = {
            "Symptom": label,
            "disease": disease,
            "confidence_score": float(prob),  # FROM MODEL
            "report": report,
            "specialist": specialist,  # FROM CONDITION MAP - NOT FROM GROQ
            "findings": [[lbl, float(p)] for lbl, p in predictions],  # FROM MODEL
            "recommendations": sections["recommendations"],
            "suggested_tests": sections["suggested_tests"],
            "predictions": [[lbl, float(p)] for lbl, p in predictions]
        }
        
        print(f"🔄 Final response specialist: {response_data['specialist']}")
        
        latest_reports["xray"] = response_data
        return JSONResponse(content=response_data)
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


# CT 2D and 3D routes handled later with full report generation

@app.post("/predict/mri/3d/")
async def generate_report_mri3d(file: UploadFile = File(...)):  
    temp_path = f"temp_mri3d_{file.filename}"
    
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

        # Generate report with TEXT-ONLY prompt (no images to Groq/Llama)
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
        
        # Define condition-to-specialist mapping for CT
        ct_specialist_map = {
            "Tumor": "Oncologist",
            "No Tumor": "Radiologist",
        }
        
        # ALWAYS use condition-based specialist - don't rely on Groq
        specialist = ct_specialist_map.get(label, "Radiologist")
        print(f"🎯 CT 2D Specialist: {label} → {specialist}")
        
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


@app.post("/predict/ct/3d/")
async def generate_report_ct3d(file: UploadFile = File(...)):
    """CT 3D volume analysis endpoint - accepts .nii, .nii.gz, .dcm files"""
    temp_path = f"temp_ct3d_{file.filename}"
    
    # Validate file format
    supported_formats = ['.nii', '.nii.gz', '.dcm']
    file_ext = PathLib(file.filename).suffix.lower()
    if file_ext not in supported_formats and not file.filename.endswith('.nii.gz'):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(supported_formats)}"
        )
    
    with open(temp_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    
    try:
        # Get predictions from CT 3D model
        raw_preds = process_ct(temp_path, mode="3d", device="cpu")
        # raw_preds is list of tuples: [("No Tumor", 0.45), ("Tumor", 0.55), ("Label", "Tumor")]
        
        # Extract label and probabilities
        tumor_label = next((item[1] for item in raw_preds if item[0] == "Label"), "Indeterminate")
        tumor_prob = next((item[1] for item in raw_preds if item[0] == "Tumor"), 0.5)
        no_tumor_prob = next((item[1] for item in raw_preds if item[0] == "No Tumor"), 0.5)
        
        # Ensure all values are valid floats
        tumor_prob = float(tumor_prob) if not np.isnan(tumor_prob) else 0.5
        no_tumor_prob = float(no_tumor_prob) if not np.isnan(no_tumor_prob) else 0.5
        
        print(f"[CT3D] Predictions: {raw_preds}")
        print(f"[CT3D] Top Prediction: {tumor_label}, Confidence: {tumor_prob*100:.2f}%")
        
        os.remove(temp_path)
        
        # Format predictions for prompt
        all_predictions_str = "\n".join([f"- {lbl}: {p*100:.2f}%" for lbl, p in raw_preds if lbl != "Label"])
        
        prompt = f"""
You are a medical specialist in interpreting 3D CT scan results.

The AI model has analyzed the 3D CT volume and provided the following analysis:
{all_predictions_str}

Final Assessment: {tumor_label} with {tumor_prob*100:.2f}% confidence.

Generate a structured medical report with the following sections:

1. **SPECIALIST**: Based on the diagnosis of {tumor_label}, recommend the MOST APPROPRIATE medical specialist the patient should consult. Provide ONLY the specialist name (e.g., "Oncologist", "Radiologist", "General Physician").

2. **Explanation**: Provide a clear, compassionate explanation of what {tumor_label} means for the patient in 2-3 sentences. Explain how 3D CT scans help with volumetric analysis and detection.

3. **RECOMMENDATIONS**: Provide EXACTLY 3 disorder-specific clinical recommendations based on {tumor_label}. These should be:
   - Specific to {tumor_label} management and treatment
   - Actionable next steps for the patient
   - Include specialist consultation if needed

4. **SUGGESTED TESTS**: Suggest EXACTLY 3 diagnostic tests or procedures specific to {tumor_label}:
   - Confirmatory tests for {tumor_label}
   - Additional imaging if relevant
   - Biopsy or other procedures specific to this diagnosis

5. **DISCLAIMER**: Include a standard medical AI disclaimer

Format your response EXACTLY as follows:

Condition Detected: {tumor_label}

SPECIALIST: [Specialist name only]

[2-3 sentence explanation paragraph about what this means and how 3D CT helps]

RECOMMENDATIONS:
1. [Specific recommendation for {tumor_label}]
2. [Specific recommendation for {tumor_label}]
3. [Specific recommendation for {tumor_label}]

SUGGESTED TESTS:
1. [Specific test for {tumor_label}]
2. [Specific test for {tumor_label}]
3. [Specific test for {tumor_label}]

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
        disease = match.group(1).strip() if match else tumor_label
        
        # Define condition-to-specialist mapping for CT 3D
        ct_specialist_map = {
            "Tumor": "Oncologist",
            "No Tumor": "Radiologist",
            "Indeterminate": "Radiologist"
        }
        
        # Use condition-based specialist
        specialist = ct_specialist_map.get(tumor_label, "Radiologist")
        print(f"[CT3D] Specialist: {tumor_label} → {specialist}")
        
        # Parse structured sections
        sections = parse_report_sections(report)

        # Build response
        latest_reports["ct3d"] = {
            "Symptom": tumor_label,
            "disease": disease,
            "confidence_score": float(tumor_prob),
            "report": report,
            "specialist": specialist,
            "findings": [
                ["No Tumor", float(no_tumor_prob)],
                ["Tumor", float(tumor_prob)],
                ["Assessment", float(tumor_prob)]
            ],
            "recommendations": sections["recommendations"],
            "suggested_tests": sections["suggested_tests"],
            "predictions": [
                ["No Tumor", float(no_tumor_prob)],
                ["Tumor", float(tumor_prob)],
                ["Assessment", float(tumor_prob)]
            ]
        }
        return JSONResponse(latest_reports["ct3d"])
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        print(f"[CT3D] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/predict/{modality}/")
async def debug_predict(modality: str = Path(...), file: UploadFile = File(...)):
    """Debug endpoint to see raw model predictions without Groq processing"""
    modality = modality.lower()
    
    if modality not in ["xray", "ct", "mri", "ultrasound"]:
        raise HTTPException(status_code=400, detail="Invalid modality. Must be one of: xray, ct, mri, ultrasound")
    
    temp_path = f"temp_debug_{modality}_{file.filename}"
    
    try:
        with open(temp_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
        
        # Get raw predictions
        if modality == "xray":
            predictions = process_xray(temp_path, device="cpu")
        elif modality == "ct":
            predictions = process_ct(temp_path, mode="2d", device="cpu")
        elif modality == "mri":
            predictions = process_mri(temp_path, mode="3d", device="cpu", top_k=4)
        elif modality == "ultrasound":
            predictions = process_ultrasound(temp_path, device="cpu", top_k=5)
        
        os.remove(temp_path)
        
        print(f"\n{'='*60}")
        print(f"DEBUG: {modality.upper()} Raw Predictions")
        print(f"{'='*60}")
        for label, prob in predictions:
            print(f"  {label:20s} {prob:6.2%}")
        print(f"{'='*60}\n")
        
        return {
            "modality": modality,
            "predictions": [[label, float(prob)] for label, prob in predictions],
            "top_prediction": {
                "label": predictions[0][0],
                "confidence": float(predictions[0][1])
            } if predictions else None
        }
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")


# Chat endpoints
class ChatRequest(BaseModel):
    message: str
    report_context: Optional[dict] = None


@app.post("/chat_with_report/")
async def chat_with_report(request: ChatRequest):
    """
    Chat with AI about a medical report
    """
    try:
        context = ""
        if request.report_context:
            report = request.report_context.get("report", "")
            disease = request.report_context.get("disease", "")
            context = f"\n\nReport context:\nDisease: {disease}\nReport: {report[:500]}..."  # Limit context size
        
        prompt = f"""You are a helpful medical AI assistant. Answer questions about medical reports clearly and compassionately.
{context}

User question: {request.message}

Provide a clear, accurate response. If you cannot provide medical advice, suggest consulting a healthcare professional."""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512
        )
        
        return {
            "response": response.choices[0].message.content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post("/public-chat/")
async def public_chat(request: ChatRequest):
    """
    General medical chat without report context
    """
    try:
        prompt = f"""You are a helpful medical information AI assistant. Answer general medical questions clearly and compassionately.
Always remind users to consult with healthcare professionals for diagnosis or treatment.

User question: {request.message}

Provide a clear, accurate response about medical information."""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512
        )
        
        return {
            "response": response.choices[0].message.content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


