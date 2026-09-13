from flask import (
    Flask,
    render_template,
    request,
    session,
    redirect,
    url_for,
    send_file
)

import pandas as pd
import numpy as np
import joblib
import os
import io

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.secret_key = "cvd-xai-research-secret-key"


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "xgb_model.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = joblib.load(MODEL_PATH)

    print("======================================")
    print("XGBoost model loaded successfully")
    print("Model path:", MODEL_PATH)
    print("======================================")

except Exception as e:

    print("======================================")
    print("ERROR LOADING MODEL")
    print("======================================")
    print(e)

    model = None


# ============================================================
# FEATURE ORDER
# ============================================================

FEATURE_COLUMNS = [

    "age_years",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "bmi_calculated",
    "pulse_pressure",
    "lifestyle_risk_score"

]


# ============================================================
# READABLE FEATURE NAMES
# ============================================================

FEATURE_LABELS = {

    "age_years":
        "Age",

    "gender":
        "Gender",

    "height":
        "Height",

    "weight":
        "Weight",

    "ap_hi":
        "Systolic Blood Pressure",

    "ap_lo":
        "Diastolic Blood Pressure",

    "cholesterol":
        "Cholesterol",

    "gluc":
        "Glucose Level",

    "smoke":
        "Smoking",

    "alco":
        "Alcohol Consumption",

    "active":
        "Physical Activity",

    "bmi_calculated":
        "BMI",

    "pulse_pressure":
        "Pulse Pressure",

    "lifestyle_risk_score":
        "Lifestyle Risk Score"

}


def readable_feature_name(feature):

    return FEATURE_LABELS.get(
        feature,
        feature.replace("_", " ").title()
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(data):

    # --------------------------------------------------------
    # Height conversion
    # --------------------------------------------------------

    height_m = data["height"] / 100.0

    if height_m <= 0:

        raise ValueError(
            "Height must be greater than zero."
        )


    # --------------------------------------------------------
    # BMI
    # --------------------------------------------------------

    bmi = (
        data["weight"] /
        (height_m ** 2)
    )


    # --------------------------------------------------------
    # Pulse Pressure
    # --------------------------------------------------------

    pulse_pressure = (
        data["ap_hi"] -
        data["ap_lo"]
    )


    # --------------------------------------------------------
    # Lifestyle Risk Score
    # --------------------------------------------------------

    lifestyle_risk_score = (

        data["smoke"]
        +
        data["alco"]
        +
        (1 - data["active"])

    )


    # --------------------------------------------------------
    # Create feature DataFrame
    # --------------------------------------------------------

    features = pd.DataFrame({

        "age_years": [
            data["age_years"]
        ],

        "gender": [
            data["gender"]
        ],

        "height": [
            data["height"]
        ],

        "weight": [
            data["weight"]
        ],

        "ap_hi": [
            data["ap_hi"]
        ],

        "ap_lo": [
            data["ap_lo"]
        ],

        "cholesterol": [
            data["cholesterol"]
        ],

        "gluc": [
            data["gluc"]
        ],

        "smoke": [
            data["smoke"]
        ],

        "alco": [
            data["alco"]
        ],

        "active": [
            data["active"]
        ],

        "bmi_calculated": [
            bmi
        ],

        "pulse_pressure": [
            pulse_pressure
        ],

        "lifestyle_risk_score": [
            lifestyle_risk_score
        ]

    })


    # --------------------------------------------------------
    # Ensure exact feature order
    # --------------------------------------------------------

    features = features[
        FEATURE_COLUMNS
    ]


    return features


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(probability):

    if probability < 0.25:

        return "Low"

    elif probability < 0.50:

        return "Moderate"

    elif probability < 0.75:

        return "High"

    else:

        return "Very High"


# ============================================================
# SHAP EXPLANATION
# ============================================================

def generate_shap_explanation(features):

    """
    Generate SHAP explanations for the current patient.

    Returns a list containing:

        feature_name
        feature
        shap_value
        impact
    """

    explanation = []


    if model is None:

        return explanation


    try:

        import shap


        # ----------------------------------------------------
        # Create TreeExplainer
        # ----------------------------------------------------

        explainer = shap.TreeExplainer(
            model
        )


        # ----------------------------------------------------
        # Calculate SHAP values
        # ----------------------------------------------------

        shap_values = explainer.shap_values(
            features
        )


        # ----------------------------------------------------
        # Handle different SHAP/XGBoost output formats
        # ----------------------------------------------------

        if isinstance(shap_values, list):

            # Binary classification
            if len(shap_values) > 1:

                shap_values = shap_values[1]

            else:

                shap_values = shap_values[0]


        shap_values = np.asarray(
            shap_values
        )


        # ----------------------------------------------------
        # Flatten one-patient result
        # ----------------------------------------------------

        if shap_values.ndim == 2:

            values = shap_values[0]

        elif shap_values.ndim == 1:

            values = shap_values

        else:

            values = shap_values.reshape(-1)


        # ----------------------------------------------------
        # XGBoost newer output can occasionally contain
        # one additional/base-value dimension.
        # Keep only feature-length values.
        # ----------------------------------------------------

        if len(values) != len(FEATURE_COLUMNS):

            values = values[:len(FEATURE_COLUMNS)]


        # ----------------------------------------------------
        # Build explanation
        # ----------------------------------------------------

        for feature, value in zip(
            FEATURE_COLUMNS,
            values
        ):

            value = float(value)


            if value > 0:

                impact = "Increases Risk"

            elif value < 0:

                impact = "Reduces Risk"

            else:

                impact = "Minimal Impact"


            explanation.append({

                "feature":
                    feature,

                "feature_name":
                    readable_feature_name(
                        feature
                    ),

                "shap_value":
                    value,

                "impact":
                    impact

            })


        # ----------------------------------------------------
        # Sort by absolute SHAP contribution
        # ----------------------------------------------------

        explanation.sort(

            key=lambda x:
            abs(x["shap_value"]),

            reverse=True

        )


        return explanation


    except Exception as e:

        print("======================================")
        print("SHAP ERROR")
        print("======================================")
        print(e)

        return []


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION PAGE + PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
def predict():

    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    if request.method == "GET":

        return render_template(
            "predict.html"
        )


    # --------------------------------------------------------
    # POST
    # --------------------------------------------------------

    try:

        data = {

            "age_years":
                int(
                    request.form["age_years"]
                ),

            "gender":
                int(
                    request.form["gender"]
                ),

            "height":
                float(
                    request.form["height"]
                ),

            "weight":
                float(
                    request.form["weight"]
                ),

            "ap_hi":
                float(
                    request.form["ap_hi"]
                ),

            "ap_lo":
                float(
                    request.form["ap_lo"]
                ),

            "cholesterol":
                int(
                    request.form["cholesterol"]
                ),

            "gluc":
                int(
                    request.form["gluc"]
                ),

            "smoke":
                int(
                    request.form["smoke"]
                ),

            "alco":
                int(
                    request.form["alco"]
                ),

            "active":
                int(
                    request.form["active"]
                )

        }


    except Exception as e:

        return f"""
        <h2>Invalid patient information</h2>
        <p>{e}</p>
        <a href="/predict">
            Return to assessment
        </a>
        """


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if data["height"] <= 0:

        return """
        <h2>Invalid height</h2>
        <p>Height must be greater than zero.</p>
        <a href="/predict">
            Return to assessment
        </a>
        """


    if data["weight"] <= 0:

        return """
        <h2>Invalid weight</h2>
        <p>Weight must be greater than zero.</p>
        <a href="/predict">
            Return to assessment
        </a>
        """


    # --------------------------------------------------------
    # Create engineered features
    # --------------------------------------------------------

    try:

        features = create_features(
            data
        )

    except Exception as e:

        return f"""
        <h2>Feature Engineering Error</h2>
        <p>{e}</p>
        <a href="/predict">
            Return to assessment
        </a>
        """


    print("\n======================================")
    print("PATIENT FEATURES")
    print("======================================")

    print(features)


    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if model is None:

        return """
        <h2>Model could not be loaded.</h2>

        <p>
        Please check:
        models/xgb_model.pkl
        </p>
        """


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        prediction = int(
            model.predict(
                features
            )[0]
        )


        probability = float(

            model.predict_proba(
                features
            )[0][1]

        )


    except Exception as e:

        print("Prediction error:")
        print(e)

        return f"""
        <h2>Prediction Error</h2>

        <p>{e}</p>

        <p>
        Expected features:
        {FEATURE_COLUMNS}
        </p>

        <a href="/predict">
            Return to assessment
        </a>
        """


    # --------------------------------------------------------
    # Risk classification
    # --------------------------------------------------------

    risk_level = classify_risk(
        probability
    )


    # --------------------------------------------------------
    # SHAP explanation
    # --------------------------------------------------------

    explanation = generate_shap_explanation(
        features
    )


    # --------------------------------------------------------
    # Store patient data
    # --------------------------------------------------------

    session["patient"] = data


    session["features"] = (
        features
        .to_dict(
            orient="records"
        )[0]
    )


    session["probability"] = (
        probability
    )


    session["prediction"] = (
        prediction
    )


    session["risk_level"] = (
        risk_level
    )


    session["explanation"] = (
        explanation
    )


    # --------------------------------------------------------
    # Redirect to results
    # --------------------------------------------------------

    return redirect(
        url_for(
            "results"
        )
    )


# ============================================================
# RESULTS PAGE
# ============================================================

@app.route("/results")
def results():

    if "probability" not in session:

        return redirect(
            url_for("index")
        )


    return render_template(

        "results.html",

        probability=round(
            session["probability"] * 100,
            2
        ),

        risk_level=session.get(
            "risk_level",
            "Unknown"
        ),

        prediction=session.get(
            "prediction",
            0
        ),

        patient=session.get(
            "patient",
            {}
        )

    )


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

@app.route("/explainability")
def explainability():

    if "features" not in session:

        return redirect(
            url_for("index")
        )


    explanation = session.get(
        "explanation",
        []
    )


    # --------------------------------------------------------
    # Safety fallback:
    # regenerate explanation if missing
    # --------------------------------------------------------

    if not explanation:

        features = pd.DataFrame([
            session["features"]
        ])

        explanation = generate_shap_explanation(
            features
        )

        session["explanation"] = explanation


    return render_template(

        "explainability.html",

        explanation=explanation

    )


# ============================================================
# WHAT-IF ANALYSIS
# ============================================================

@app.route(
    "/what-if",
    methods=["GET", "POST"]
)
@app.route(
    "/counterfactual",
    methods=["GET", "POST"]
)
def what_if():

    if "patient" not in session:

        return redirect(
            url_for("index")
        )


    original_patient = (
        session["patient"]
    )

    result = None


    # --------------------------------------------------------
    # POST
    # --------------------------------------------------------

    if request.method == "POST":

        try:

            modified_data = (
                original_patient.copy()
            )


            modified_data["ap_hi"] = float(
                request.form["ap_hi"]
            )


            modified_data["ap_lo"] = float(
                request.form["ap_lo"]
            )


            modified_data["weight"] = float(
                request.form["weight"]
            )


            modified_data["smoke"] = int(
                request.form["smoke"]
            )


            modified_data["active"] = int(
                request.form["active"]
            )


            # ------------------------------------------------
            # Create new features
            # ------------------------------------------------

            features = create_features(
                modified_data
            )


            # ------------------------------------------------
            # New prediction
            # ------------------------------------------------

            new_probability = float(

                model.predict_proba(
                    features
                )[0][1]

            )


            original_probability = (
                session["probability"]
            )


            # ------------------------------------------------
            # Change
            #
            # Positive = risk increased
            # Negative = risk decreased
            # ------------------------------------------------

            change = (

                new_probability
                -
                original_probability

            )


            result = {

                "original": round(
                    original_probability * 100,
                    2
                ),

                "new": round(
                    new_probability * 100,
                    2
                ),

                "change": round(
                    change * 100,
                    2
                ),

                "risk_level":
                    classify_risk(
                        new_probability
                    )

            }


        except Exception as e:

            print(
                "What-if analysis error:",
                e
            )

            result = {

                "original":
                    round(
                        session["probability"] * 100,
                        2
                    ),

                "new":
                    "Error",

                "change":
                    "Error",

                "risk_level":
                    str(e)

            }


    return render_template(

        "what_if.html",

        patient=original_patient,

        result=result

    )


# ============================================================
# PERSONALISED RECOMMENDATIONS
# ============================================================

@app.route("/recommendations")
def recommendations():

    if "patient" not in session:

        return redirect(
            url_for("index")
        )


    patient = session["patient"]

    recommendations = []


    # --------------------------------------------------------
    # Blood Pressure
    # --------------------------------------------------------

    if patient["ap_hi"] >= 140:

        recommendations.append({

            "title":
                "Blood Pressure Monitoring",

            "text":
                "The recorded systolic blood pressure is elevated. "
                "Regular blood-pressure monitoring and discussion "
                "with a qualified healthcare professional may be "
                "appropriate."

        })


    # --------------------------------------------------------
    # BMI
    # --------------------------------------------------------

    bmi = (

        patient["weight"]
        /
        ((patient["height"] / 100) ** 2)

    )


    if bmi >= 25:

        recommendations.append({

            "title":
                "Weight Management",

            "text":
                "The calculated BMI is above the healthy range. "
                "Appropriate nutrition, physical activity and "
                "healthy weight-management strategies may be "
                "considered with professional guidance."

        })


    # --------------------------------------------------------
    # Smoking
    # --------------------------------------------------------

    if patient["smoke"] == 1:

        recommendations.append({

            "title":
                "Smoking Cessation",

            "text":
                "Smoking is a modifiable cardiovascular risk factor. "
                "Evidence-based smoking cessation support may help "
                "reduce cardiovascular risk."

        })


    # --------------------------------------------------------
    # Physical Activity
    # --------------------------------------------------------

    if patient["active"] == 0:

        recommendations.append({

            "title":
                "Physical Activity",

            "text":
                "The patient is recorded as physically inactive. "
                "An appropriate physical activity plan can be "
                "discussed with a qualified healthcare professional."

        })


    # --------------------------------------------------------
    # Cholesterol
    # --------------------------------------------------------

    if patient["cholesterol"] >= 2:

        recommendations.append({

            "title":
                "Cholesterol Management",

            "text":
                "The recorded cholesterol category is above normal. "
                "Further cardiovascular risk assessment and "
                "cholesterol management should be discussed with "
                "a qualified healthcare professional."

        })


    # --------------------------------------------------------
    # Glucose
    # --------------------------------------------------------

    if patient["gluc"] >= 2:

        recommendations.append({

            "title":
                "Glucose Monitoring",

            "text":
                "The recorded glucose category is above normal. "
                "Appropriate monitoring and professional clinical "
                "assessment may be considered."

        })


    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    if not recommendations:

        recommendations.append({

            "title":
                "General Cardiovascular Health",

            "text":
                "Continue healthy lifestyle practices and regular "
                "cardiovascular health monitoring. Individual "
                "clinical advice should be obtained from a qualified "
                "healthcare professional."

        })


    return render_template(

        "recommendations.html",

        recommendations=
            recommendations

    )


# ============================================================
# CLINICAL REPORT
# ============================================================

@app.route("/report")
def report():

    if "patient" not in session:

        return redirect(
            url_for("index")
        )


    explanation = session.get(
        "explanation",
        []
    )


    return render_template(

        "report.html",

        patient=session[
            "patient"
        ],

        probability=round(

            session[
                "probability"
            ] * 100,

            2

        ),

        risk_level=session[
            "risk_level"
        ],

        explanation=explanation

    )


# ============================================================
# RESET SESSION
# ============================================================

@app.route("/reset")
def reset():

    session.clear()

    return redirect(
        url_for("index")
    )


# ============================================================
# DOWNLOAD PDF REPORT
# ============================================================

@app.route("/download-report")
def download_report():

    # --------------------------------------------------------
    # Check assessment
    # --------------------------------------------------------

    if "patient" not in session:

        return redirect(
            url_for("index")
        )


    patient = session["patient"]

    probability = (
        session.get(
            "probability",
            0
        ) * 100
    )

    risk_level = session.get(
        "risk_level",
        "Unknown"
    )

    prediction = session.get(
        "prediction",
        0
    )

    explanation = session.get(
        "explanation",
        []
    )


    # --------------------------------------------------------
    # Generate SHAP explanation if missing
    # --------------------------------------------------------

    if not explanation:

        features = pd.DataFrame([
            session["features"]
        ])

        explanation = generate_shap_explanation(
            features
        )


    # --------------------------------------------------------
    # Create PDF in memory
    # --------------------------------------------------------

    buffer = io.BytesIO()


    document = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=40,

        leftMargin=40,

        topMargin=40,

        bottomMargin=40

    )


    styles = getSampleStyleSheet()


    title_style = styles["Title"]

    title_style.alignment = TA_CENTER


    heading_style = styles["Heading2"]

    normal_style = styles["BodyText"]


    story = []


    # ========================================================
    # TITLE
    # ========================================================

    story.append(

        Paragraph(

            "Cardiovascular Disease Risk Assessment Report",

            title_style

        )

    )


    story.append(
        Spacer(1, 10)
    )


    story.append(

        Paragraph(

            "Explainable AI Clinical Decision Support System",

            normal_style

        )

    )


    story.append(
        Spacer(1, 25)
    )


    # ========================================================
    # PATIENT INFORMATION
    # ========================================================

    story.append(

        Paragraph(

            "1. Patient Information",

            heading_style

        )

    )


    gender = (

        "Female"

        if patient["gender"] == 1

        else

        "Male"

    )


    smoking = (

        "Yes"

        if patient["smoke"] == 1

        else

        "No"

    )


    alcohol = (

        "Yes"

        if patient["alco"] == 1

        else

        "No"

    )


    activity = (

        "Active"

        if patient["active"] == 1

        else

        "Inactive"

    )


    cholesterol_labels = {

        1: "Normal",

        2: "Above Normal",

        3: "Well Above Normal"

    }


    glucose_labels = {

        1: "Normal",

        2: "Above Normal",

        3: "Well Above Normal"

    }


    patient_table = [

        ["Parameter", "Value"],

        [
            "Age",
            f'{patient["age_years"]} years'
        ],

        [
            "Gender",
            gender
        ],

        [
            "Height",
            f'{patient["height"]:.1f} cm'
        ],

        [
            "Weight",
            f'{patient["weight"]:.1f} kg'
        ],

        [
            "Systolic Blood Pressure",
            f'{patient["ap_hi"]:.0f} mmHg'
        ],

        [
            "Diastolic Blood Pressure",
            f'{patient["ap_lo"]:.0f} mmHg'
        ],

        [
            "Cholesterol",
            cholesterol_labels.get(
                patient["cholesterol"],
                str(patient["cholesterol"])
            )
        ],

        [
            "Glucose",
            glucose_labels.get(
                patient["gluc"],
                str(patient["gluc"])
            )
        ],

        [
            "Smoking",
            smoking
        ],

        [
            "Alcohol Consumption",
            alcohol
        ],

        [
            "Physical Activity",
            activity
        ]

    ]


    table = Table(

        patient_table,

        colWidths=[
            200,
            260
        ]

    )


    table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#123b5d")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                8
            )

        ])

    )


    story.append(table)


    story.append(
        Spacer(1, 25)
    )


    # ========================================================
    # RISK PREDICTION
    # ========================================================

    story.append(

        Paragraph(

            "2. Cardiovascular Risk Prediction",

            heading_style

        )

    )


    prediction_text = (

        "Cardiovascular disease risk detected"

        if prediction == 1

        else

        "Lower cardiovascular disease risk predicted"

    )


    prediction_table = [

        ["Assessment", "Result"],

        [
            "Prediction",
            prediction_text
        ],

        [
            "Probability",
            f"{probability:.2f}%"
        ],

        [
            "Risk Level",
            risk_level
        ]

    ]


    table = Table(

        prediction_table,

        colWidths=[
            200,
            260
        ]

    )


    table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#123b5d")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                8
            )

        ])

    )


    story.append(table)


    story.append(
        Spacer(1, 25)
    )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    story.append(

        Paragraph(

            "3. Explainable AI Analysis",

            heading_style

        )

    )


    story.append(

        Paragraph(

            "The following features had the largest absolute "
            "SHAP contributions to the individual prediction. "
            "Positive SHAP values indicate a contribution toward "
            "higher predicted risk, while negative values indicate "
            "a contribution toward lower predicted risk.",

            normal_style

        )

    )


    story.append(
        Spacer(1, 10)
    )


    shap_table = [

        [
            "Feature",
            "SHAP Value",
            "Impact"
        ]

    ]


    for item in explanation[:10]:

        shap_value = float(
            item["shap_value"]
        )


        if shap_value > 0:

            direction = "Increases risk"

        elif shap_value < 0:

            direction = "Reduces risk"

        else:

            direction = "Minimal impact"


        shap_table.append(

            [

                readable_feature_name(
                    item["feature"]
                ),

                f"{shap_value:.4f}",

                direction

            ]

        )


    if len(shap_table) == 1:

        shap_table.append(

            [
                "No SHAP explanation available",
                "-",
                "-"
            ]

        )


    table = Table(

        shap_table,

        colWidths=[
            200,
            120,
            140
        ]

    )


    table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#123b5d")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7
            )

        ])

    )


    story.append(table)


    story.append(
        Spacer(1, 25)
    )


    # ========================================================
    # CLINICAL DISCLAIMER
    # ========================================================

    story.append(

        Paragraph(

            "Clinical Disclaimer",

            heading_style

        )

    )


    story.append(

        Paragraph(

            "This report is generated by an experimental "
            "machine-learning Clinical Decision Support System "
            "developed for MSc dissertation research. The "
            "predicted probability, risk classification and "
            "explanations are intended to support, not replace, "
            "professional clinical judgement. The output should "
            "not be considered a medical diagnosis or treatment "
            "recommendation.",

            normal_style

        )

    )


    # ========================================================
    # BUILD PDF
    # ========================================================

    document.build(
        story
    )


    buffer.seek(0)


    return send_file(

        buffer,

        as_attachment=True,

        download_name=(
            "CVD_Risk_Assessment_Report.pdf"
        ),

        mimetype="application/pdf"

    )


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
