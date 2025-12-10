# New Endpoints for Ultrasound and CT 2D
# Add these to main.py after line 639

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