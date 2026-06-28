import logging
import json
import pandas as pd
from typing import Dict, Any, Tuple
from app.config import settings

logger = logging.getLogger(__name__)

# Try importing google.generativeai. If not installed or any import error, we fallback.
try:
    import google.generativeai as genai
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

def get_gemini_client():
    if not HAS_GEMINI_SDK or not settings.GEMINI_API_KEY:
        return None
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        # We use the standard recommended model
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model
    except Exception as e:
        logger.warning(f"Failed to configure Gemini SDK: {e}")
        return None

def fallback_heuristic_analyzer(df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
    """
    Local rule-based heuristic profiling when Gemini API is unavailable.
    """
    columns = list(df.columns)
    
    # Simple target deduction
    if not target_col or target_col not in df.columns:
        # Avoid columns named 'id', 'uuid', 'index' as target
        candidate_cols = [c for c in columns if c.lower() not in ["id", "uuid", "index", "key"]]
        target_col = candidate_cols[-1] if candidate_cols else columns[-1]

    # Deduce problem type
    target_series = df[target_col].dropna()
    unique_count = target_series.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(target_series)
    
    if not is_numeric or target_series.dtype == 'bool' or unique_count < 15:
        problem_type = "classification"
    else:
        problem_type = "regression"

    # Profile columns
    column_analysis = {}
    for col in columns:
        col_series = df[col]
        null_count = int(col_series.isnull().sum())
        unique_vals = col_series.nunique()
        dtype_str = str(col_series.dtype)
        
        # Check if ID or useless column
        is_id = col.lower() in ["id", "uuid", "index", "key", "pk", "fk"]
        is_high_cardinality_non_numeric = (not pd.api.types.is_numeric_dtype(col_series)) and (unique_vals > len(df) * 0.9)
        
        if col == target_col:
            role = "target"
            explanation = "Identified as the model target column."
        elif is_id or is_high_cardinality_non_numeric:
            role = "ignore"
            explanation = "Useless ID or high-cardinality non-numeric identifier."
        else:
            role = "feature"
            explanation = f"Predictive feature of type {dtype_str}."

        feat_type = "numerical" if pd.api.types.is_numeric_dtype(col_series) else "categorical"
        
        column_analysis[col] = {
            "role": role,
            "type": feat_type,
            "explanation": explanation
        }

    # Generic description
    description = (
        f"Rule-based Profile: This dataset contains {len(df)} rows and {len(columns)} columns. "
        f"The AutoML engine has identified it as a {problem_type} task targeting '{target_col}'."
    )

    return {
        "description": description,
        "suggested_target": target_col,
        "suggested_problem_type": problem_type,
        "column_analysis": column_analysis
    }

def analyze_dataset_with_ai(df: pd.DataFrame, user_target: str = None) -> Dict[str, Any]:
    """
    Analyzes dataset columns, sample data, and metadata using Gemini.
    Falls back to a local heuristic analyzer if Gemini is not configured or fails.
    """
    model = get_gemini_client()
    if not model:
        logger.info("Gemini API not configured or SDK unavailable. Running heuristic analyzer.")
        return fallback_heuristic_analyzer(df, user_target)

    try:
        # Prepare schema details for the prompt
        columns_summary = []
        for col in df.columns:
            series = df[col]
            null_count = int(series.isnull().sum())
            unique_count = series.nunique()
            dtype = str(series.dtype)
            sample_vals = series.dropna().head(3).tolist()
            columns_summary.append({
                "column_name": col,
                "data_type": dtype,
                "unique_values": unique_count,
                "null_count": null_count,
                "sample_values": sample_vals
            })

        # Fetch sample rows
        sample_rows = df.head(5).to_dict(orient="records")
        
        prompt = f"""
You are an expert MLOps dataset profiling agent. Analyze the metadata and sample rows of the dataset below.

Dataset Columns Metadata:
{json.dumps(columns_summary, indent=2)}

First 5 Sample Rows of the Dataset:
{json.dumps(sample_rows, indent=2)}

User Specified Target Column (optional): {user_target or 'None'}

Please perform the following profiling tasks:
1. Domain Identification: Identify what domain/field this dataset represents and write a clear 2-3 sentence description of what the dataset is about.
2. Target Column Suggestion: Identify the single most logical target column to predict (use the User Specified Target Column if it is valid, otherwise deduce the best target).
3. Problem Type: Determine whether predicting the target is a "classification" or "regression" problem.
4. Column Roles Classification: For every single column, specify:
   - role: "target", "feature", or "ignore" (use "ignore" for primary keys, row IDs, timestamps, or unique hashes that hold no predictive power).
   - type: "numerical" or "categorical".
   - explanation: A short, 5-10 word explanation of why this role and type were assigned.

You MUST respond with a single, raw, valid JSON object ONLY, matching this schema EXACTLY:
{{
  "description": "Domain description of the dataset...",
  "suggested_target": "name_of_target_column",
  "suggested_problem_type": "classification",
  "column_analysis": {{
    "column_name": {{
      "role": "feature",
      "type": "numerical",
      "explanation": "Brief explanation of feature type..."
    }}
  }}
}}
Do NOT wrap the JSON inside markdown code blocks (e.g., do not use ```json). Return only the JSON string.
"""
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Clean up in case Gemini wraps it in code blocks anyway
        if text.startswith("```json"):
            text = text.replace("```json", "", 1)
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        analysis_result = json.loads(text)
        
        # Validate critical fields are present
        if "description" not in analysis_result or "suggested_target" not in analysis_result:
            raise ValueError("Required fields missing from Gemini JSON response.")

        logger.info("Successfully profiled dataset using Gemini API.")
        return analysis_result

    except Exception as e:
        logger.error(f"Gemini profiling failed: {e}. Falling back to heuristic analyzer.", exc_info=True)
        return fallback_heuristic_analyzer(df, user_target)
