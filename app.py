# =========================================================
# Student Retention Agent
# =========================================================
# This Streamlit prototype implements the system described in the final report:
# "From Prediction to Support: A Human-in-the-Loop Student Retention Agent".
#
# Important evaluation framing:
# - The dataset is simulated for privacy and reproducibility.
# - The model is a supervised Random Forest trained on structured student records.
# - LangGraph coordinates the workflow as a single tool-using agent.
# - The system moves beyond a static dashboard by predicting, explaining,
#   recommending, safety-checking, messaging, simulating feedback, updating state,
#   and re-predicting when an intervention is actually applied.
# - Low-risk, high-confidence students receive no message and no database update.
# - Sensitive, high-risk, or low-confidence cases are routed to human review.
# - The batch triage tab demonstrates how the same logic can scan all students
#   and produce a prioritized support queue for staff.
#
# No real student records are used in this prototype.
# In a production deployment, the simulated data layer would be replaced with
# approved, anonymized, securely governed LMS, SIS, attendance, assessment, and
# student-support data.
# =========================================================

from typing import TypedDict, Dict, Any, List

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity

try:
    from langgraph.graph import StateGraph, END
except Exception:
    StateGraph = None
    END = None


# =========================================================
# Feature schema
# =========================================================
# These fields are the structured inputs used by the Random Forest model and
# by the agent's explanation and support-recommendation tools.
# They mirror the report's "Feature schema" table.
# The schema is intentionally simple and numeric so that the proof-of-concept
# can run locally and remain reproducible.

FEATURES = [
    "gpa",
    "attendance_rate",
    "lms_logins_week",
    "assignment_completion",
    "missed_assignments",
    "grade_trend",
    "survey_wellbeing",
    "support_requests",
]

FEATURE_EXPLANATIONS = {
    "gpa": "Academic performance on a 0 to 9 GPA-style scale.",
    "attendance_rate": "Percentage of classes attended. This simulates participation/attendance records.",
    "lms_logins_week": "Weekly LMS activity count. Inspired by VLE/LMS interaction logs.",
    "assignment_completion": "Percentage of assigned work completed or submitted.",
    "missed_assignments": "Number of missed or incomplete assignments during the monitoring period.",
    "grade_trend": "Recent change in academic performance. Negative means declining; positive means improving.",
    "survey_wellbeing": "Student wellbeing/support proxy on a 1 to 10 survey-style scale.",
    "support_requests": "Number of student support interactions or support need signals.",
}


# =========================================================
# 1. Research-informed simulated data generation
# =========================================================
# This section creates a temporary synthetic student dataset.
# It is not random in a completely arbitrary way: a set of latent risk variables
# is used to create realistic correlations across GPA, attendance, LMS activity,
# assignment completion, missed assignments, wellbeing proxy, and support signals.
# This matches the report's explanation that the data are open-data inspired,
# not real institutional records.

@st.cache_data
def generate_research_informed_student_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Creates simulated student records using a schema inspired by open learning analytics datasets.

    Why simulated:
    - Real student records are private and require institutional approval.
    - Public datasets do not usually contain GPA, attendance, LMS engagement, wellbeing,
      support requests, and intervention outcomes all in one consistent table.
    - This simulation lets the agent workflow be tested safely while reflecting common
      feature categories from learning analytics research.

    Inspiration:
    - UCI Predict Students' Dropout and Academic Success: academic path, socioeconomic
      and outcome-style dropout/graduate labels.
    - Open University Learning Analytics Dataset: student demographics, assessments,
      registrations, and virtual learning environment interaction logs.
    """

    rng = np.random.default_rng(seed)

    # Latent student risk propensity. Higher means more likely to struggle.
    # This creates realistic correlations across features rather than fully independent random columns.
    academic_risk = rng.beta(2.2, 2.8, n)
    engagement_risk = rng.beta(2.0, 3.0, n)
    wellbeing_risk = rng.beta(1.8, 3.2, n)

    # Academic performance: 0 to 9 GPA-style scale.
    gpa = (8.3 - academic_risk * 5.2 + rng.normal(0, 0.75, n)).clip(0, 9)

    # Attendance: 0 to 100 percent.
    attendance_rate = (94 - engagement_risk * 65 - wellbeing_risk * 12 + rng.normal(0, 8, n)).clip(0, 100)

    # LMS logins: weekly count inspired by VLE interaction logs.
    # Students with high engagement risk usually interact less.
    lms_mean = (16 - engagement_risk * 13 - academic_risk * 3).clip(1, 25)
    lms_logins_week = rng.poisson(lms_mean).clip(0, 35)

    # Assignment completion: percentage of assigned work submitted/completed.
    assignment_completion = (
        96 - academic_risk * 45 - engagement_risk * 25 + rng.normal(0, 10, n)
    ).clip(0, 100)

    # Missed assignments: count during the monitoring period.
    missed_lambda = (0.4 + academic_risk * 2.8 + engagement_risk * 2.2).clip(0.1, 7)
    missed_assignments = rng.poisson(missed_lambda).clip(0, 8)

    # Grade trend: negative means declining; positive means improving.
    grade_trend = (
        12 - academic_risk * 30 - engagement_risk * 12 + rng.normal(0, 8, n)
    ).clip(-40, 40)

    # Wellbeing survey-style proxy: 1 to 10.
    # Lower values represent more concerning wellbeing/support context.
    survey_wellbeing = (
        9.2 - wellbeing_risk * 5.8 - academic_risk * 0.9 + rng.normal(0, 0.9, n)
    ).clip(1, 10)

    # Support requests/support need signals.
    support_lambda = (0.4 + wellbeing_risk * 2.2 + academic_risk * 0.9 + engagement_risk * 0.6).clip(0.1, 6)
    support_requests = rng.poisson(support_lambda).clip(0, 8)

    # Hidden risk score used only to create simulated labels.
    # In real deployment, labels would come from historical retention outcomes.
    risk_score = (
        (9 - gpa) * 0.85
        + (100 - attendance_rate) * 0.035
        + (14 - lms_logins_week) * 0.12
        + (100 - assignment_completion) * 0.035
        + missed_assignments * 0.45
        + (-grade_trend) * 0.03
        + (10 - survey_wellbeing) * 0.42
        + support_requests * 0.18
        + rng.normal(0, 1.0, n)
    )

    # Low, Medium, High risk labels are created using quantiles so all classes are represented.
    labels = pd.qcut(risk_score, q=3, labels=["Low", "Medium", "High"])

    df = pd.DataFrame({
        "student_id": [f"SIM{str(i).zfill(4)}" for i in range(1, n + 1)],
        "gpa": gpa.round(2),
        "attendance_rate": attendance_rate.round(1),
        "lms_logins_week": lms_logins_week.astype(int),
        "assignment_completion": assignment_completion.round(1),
        "missed_assignments": missed_assignments.astype(int),
        "grade_trend": grade_trend.round(1),
        "survey_wellbeing": survey_wellbeing.round(1),
        "support_requests": support_requests.astype(int),
        "risk_label": labels.astype(str)
    })

    return df


# =========================================================
# 2. Random Forest model training
# =========================================================
# This is the supervised learning component of the compound AI system.
# It learns to predict the generated Low, Medium, or High risk label from the
# structured student features. The metrics displayed in the app correspond to
# the report's reported accuracy, precision, recall, and F1 values after rounding.

@st.cache_resource
def train_random_forest_model(df: pd.DataFrame):
    """
    Trains a Random Forest model.

    Rationale:
    - Student data is structured and tabular.
    - Random Forest can handle non-linear feature interactions.
    - It is robust because it averages many decision trees.
    - It provides feature importance, which supports explainability.
    """

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").astype(float)
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=7,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=9,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=7
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        preds,
        average="weighted",
        zero_division=0
    )

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    importances = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return model, metrics, importances


# =========================================================
# 3. LangGraph agent state
# =========================================================
# LangGraph passes a state object between tools.
# This TypedDict documents the information that can be produced or updated
# during one agent run: prediction, confidence, explanations, recommendations,
# safety status, generated message, updated student state, and execution trace.

class StudentAgentState(TypedDict, total=False):
    student: Dict[str, Any]
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    risk_factors: List[str]
    recommendations: List[Dict[str, Any]]
    selected_intervention: str
    safety_status: str
    safety_reason: str
    support_message: str
    message_status: str
    updated_student: Dict[str, Any]
    updated_prediction: str
    updated_confidence: float
    execution_trace: List[str]


# =========================================================
# 4. Agent tools
# =========================================================
# Each function below is a tool in the agentic workflow.
# The workflow is intentionally modular so that the system can be inspected,
# explained, and evaluated as more than a simple prediction dashboard.

# Tool 1: Predict risk.
# This uses the trained Random Forest to produce class probabilities for one
# student and selects the highest-probability class as the risk prediction.
def predict_risk_tool(state: StudentAgentState, model) -> StudentAgentState:
    student = pd.Series(state["student"])

    X_student = pd.DataFrame(
        [{feature: float(student[feature]) for feature in FEATURES}],
        columns=FEATURES
    )

    probabilities = model.predict_proba(X_student)[0]
    classes = model.classes_

    prob_dict = {c: float(p) for c, p in zip(classes, probabilities)}
    prediction = max(prob_dict, key=prob_dict.get)
    confidence = prob_dict[prediction]

    trace = state.get("execution_trace", [])
    trace.append("Tool used: Random Forest risk prediction model")

    return {
        **state,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": prob_dict,
        "execution_trace": trace
    }


# Tool 2: Explain risk factors.
# This rule-based explanation layer turns numeric feature values into readable
# reasons such as low GPA, low attendance, or missed assignments.
# It supports transparency and helps staff understand why a case was flagged.
def explain_risk_tool(state: StudentAgentState) -> StudentAgentState:
    student = pd.Series(state["student"])
    factors = []

    if student["gpa"] < 4.5:
        factors.append("low GPA")
    if student["attendance_rate"] < 65:
        factors.append("low attendance")
    if student["lms_logins_week"] < 5:
        factors.append("low LMS engagement")
    if student["assignment_completion"] < 65:
        factors.append("low assignment completion")
    if student["missed_assignments"] >= 3:
        factors.append("multiple missed assignments")
    if student["grade_trend"] < -8:
        factors.append("declining grade trend")
    if student["survey_wellbeing"] < 5:
        factors.append("low wellbeing/support score")
    if student["support_requests"] >= 3:
        factors.append("frequent support need indicators")

    if not factors:
        factors.append("a combination of moderate signals rather than one extreme factor")

    trace = state.get("execution_trace", [])
    trace.append("Tool used: risk explanation module")

    return {
        **state,
        "risk_factors": factors,
        "execution_trace": trace
    }


# Helper: Convert a student profile into a risk-needs vector.
# Larger values represent stronger need in that dimension.
# This vector lets the support recommendation layer consider the full profile
# instead of using only one isolated trigger.
def context_vector(student: pd.Series):
    return np.array([
        1 - student["gpa"] / 9,
        1 - student["attendance_rate"] / 100,
        1 - min(student["lms_logins_week"], 20) / 20,
        1 - student["assignment_completion"] / 100,
        min(student["missed_assignments"], 6) / 6,
        max(-student["grade_trend"], 0) / 40,
        1 - student["survey_wellbeing"] / 10,
        min(student["support_requests"], 6) / 6
    ]).reshape(1, -1)


# Tool 3: Recommend support.
# This maps the student's full risk-needs vector against intervention profiles.
# The result is a ranked set of support options, plus a selected top intervention.
# Low-risk and sufficiently confident cases are intentionally assigned no action.
def recommend_intervention_tool(state: StudentAgentState) -> StudentAgentState:
    student = pd.Series(state["student"])
    predicted_risk = state["prediction"]
    confidence = state["confidence"]

    trace = state.get("execution_trace", [])

    # ---------------------------------------------------------
    # New safety/action logic:
    # If the student is Low risk and the model is confident, do not recommend an intervention.
    # Continue monitoring instead of sending a support message.
    # ---------------------------------------------------------
    if predicted_risk == "Low" and confidence >= 0.60:
        trace.append("Tool used: intervention recommender selected no action because risk is low and confidence is sufficient")
        return {
            **state,
            "recommendations": [
                {
                    "intervention": "No action needed - continue monitoring",
                    "match_score": 1.0
                }
            ],
            "selected_intervention": "No action needed - continue monitoring",
            "execution_trace": trace
        }

    intervention_profiles = {
        "Academic tutoring": np.array([0.9, 0.2, 0.3, 0.8, 0.6, 0.7, 0.1, 0.2]),
        "Advisor check-in": np.array([0.4, 0.8, 0.6, 0.4, 0.5, 0.5, 0.4, 0.5]),
        "Study planning support": np.array([0.5, 0.4, 0.5, 0.7, 0.8, 0.5, 0.2, 0.4]),
        "Wellbeing support referral": np.array([0.2, 0.3, 0.3, 0.2, 0.2, 0.2, 1.0, 0.6]),
        "Peer mentoring": np.array([0.3, 0.4, 0.6, 0.3, 0.3, 0.3, 0.5, 0.4]),
        "Engagement reminder with resources": np.array([0.2, 0.7, 0.9, 0.3, 0.3, 0.2, 0.2, 0.2]),
        "Financial/support services check-in": np.array([0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 0.6, 0.9])
    }

    student_vec = context_vector(student)

    rows = []
    for name, profile in intervention_profiles.items():
        similarity = cosine_similarity(student_vec, profile.reshape(1, -1))[0][0]
        urgency_bonus = {"Low": 0.00, "Medium": 0.05, "High": 0.12}.get(predicted_risk, 0)
        score = similarity + urgency_bonus

        rows.append({
            "intervention": name,
            "match_score": round(float(score), 3)
        })

    ranked = sorted(rows, key=lambda x: x["match_score"], reverse=True)

    trace.append("Tool used: context-aware intervention recommendation engine")

    return {
        **state,
        "recommendations": ranked,
        "selected_intervention": ranked[0]["intervention"],
        "execution_trace": trace
    }


# Tool 4: Safety and human-review checker.
# This implements the report's action-control policy.
# High-risk, low-confidence, advisor, wellbeing, and financial/support cases
# are routed to human review. Low-risk, confident cases are monitored only.
def safety_checker_tool(state: StudentAgentState) -> StudentAgentState:
    predicted_risk = state["prediction"]
    intervention_name = state["selected_intervention"]
    confidence = state["confidence"]

    high_impact_interventions = {
        "Advisor check-in": "Direct staff involvement",
        "Wellbeing support referral": "Sensitive student wellbeing context",
        "Financial/support services check-in": "Sensitive personal/financial context",
    }

    review_reasons = []

    if intervention_name in high_impact_interventions:
        review_reasons.append(high_impact_interventions[intervention_name])

    if predicted_risk == "High":
        review_reasons.append("Higher-stakes decision due to high-risk prediction")

    if confidence < 0.60:
        review_reasons.append("Model uncertainty because confidence is below 0.60")

    if intervention_name == "No action needed - continue monitoring":
        safety_status = "No action needed"
        safety_reason = (
            "The student is predicted as Low risk with sufficient model confidence. "
            "No student-facing intervention is generated; the system will continue monitoring."
        )
    elif review_reasons:
        safety_status = "Human review required"
        safety_reason = (
            "Human review is required for this case. Reason(s): "
            + "; ".join(review_reasons)
            + "."
        )
    else:
        safety_status = "Safe to automate"
        safety_reason = (
            "This is a Low/Medium risk case with a low-impact intervention. "
            "It can be automated while still logging the decision."
        )

    trace = state.get("execution_trace", [])
    trace.append("Tool used: safety and human-in-the-loop checker")

    return {
        **state,
        "safety_status": safety_status,
        "safety_reason": safety_reason,
        "execution_trace": trace
    }


# Tool 5: Standardized communication generator.
# The current prototype deliberately avoids LLM-generated student messages.
# This reduces hallucination risk and keeps communication auditable.
def standardized_support_message(student_id: str, risk_factors: List[str], intervention: str, safety_status: str) -> str:
    """
    Generates a standardized, non-LLM support message.

    The safety status is used internally by the app, but it is not shown in the
    student-facing message. This keeps the message supportive and avoids exposing
    internal risk labels or review logic.
    """

    if intervention == "No action needed - continue monitoring":
        return "No student-facing message generated. The student is currently marked for monitoring only."

    return f"""Kia ora {student_id},

I’m reaching out as a friendly check-in because there may be an opportunity to support your progress with coursework and assignments.

A useful next step may be: {intervention}.

Please feel free to connect with the student support team if you would like help planning your next steps.

Kind regards,
Student Support Team
"""


# Tool 6: Attach message status to the agent state.
# The message may be queued, require staff approval, or be withheld entirely.
def generate_message_tool(state: StudentAgentState) -> StudentAgentState:
    student_id = state["student"]["student_id"]
    risk_factors = state["risk_factors"]
    intervention = state["selected_intervention"]
    safety_status = state["safety_status"]

    message = standardized_support_message(student_id, risk_factors, intervention, safety_status)

    if intervention == "No action needed - continue monitoring":
        message_status = "No message generated. Low-risk student is monitored only."
    elif safety_status == "Human review required":
        message_status = "Standardized draft generated. Staff approval is required before sending."
    else:
        message_status = "Standardized draft generated. This low-impact message can be queued for delivery."

    trace = state.get("execution_trace", [])
    trace.append("Tool used: standardized communication generator, no LLM")

    return {
        **state,
        "support_message": message,
        "message_status": message_status,
        "execution_trace": trace
    }


# Tool 7: Simulate feedback.
# This is a proof-of-concept feedback loop, not causal evidence.
# It demonstrates how the system could update state after an intervention.
def simulate_feedback_tool(state: StudentAgentState) -> StudentAgentState:
    student = pd.Series(state["student"]).copy()
    intervention = state["selected_intervention"]

    if intervention == "No action needed - continue monitoring":
        # No intervention is simulated because no action is being taken.
        trace = state.get("execution_trace", [])
        trace.append("Tool skipped: no feedback simulation needed for low-risk monitoring-only case")

        return {
            **state,
            "updated_student": student.to_dict(),
            "updated_prediction": state["prediction"],
            "updated_confidence": state["confidence"],
            "execution_trace": trace
        }

    elif intervention == "Academic tutoring":
        student["gpa"] = min(9, student["gpa"] + 0.4)
        student["assignment_completion"] = min(100, student["assignment_completion"] + 8)
        student["grade_trend"] = min(40, student["grade_trend"] + 6)

    elif intervention == "Advisor check-in":
        student["attendance_rate"] = min(100, student["attendance_rate"] + 8)
        student["lms_logins_week"] = min(35, student["lms_logins_week"] + 3)
        student["support_requests"] = max(0, student["support_requests"] - 1)

    elif intervention == "Study planning support":
        student["assignment_completion"] = min(100, student["assignment_completion"] + 10)
        student["missed_assignments"] = max(0, student["missed_assignments"] - 1)

    elif intervention == "Wellbeing support referral":
        student["survey_wellbeing"] = min(10, student["survey_wellbeing"] + 1.2)
        student["attendance_rate"] = min(100, student["attendance_rate"] + 4)

    elif intervention == "Peer mentoring":
        student["lms_logins_week"] = min(35, student["lms_logins_week"] + 3)
        student["survey_wellbeing"] = min(10, student["survey_wellbeing"] + 0.7)

    elif intervention == "Engagement reminder with resources":
        student["lms_logins_week"] = min(35, student["lms_logins_week"] + 4)
        student["attendance_rate"] = min(100, student["attendance_rate"] + 3)

    elif intervention == "Financial/support services check-in":
        student["survey_wellbeing"] = min(10, student["survey_wellbeing"] + 0.8)
        student["support_requests"] = max(0, student["support_requests"] - 1)

    trace = state.get("execution_trace", [])
    trace.append("Tool used: feedback simulator for post-intervention update")

    return {
        **state,
        "updated_student": student.to_dict(),
        "execution_trace": trace
    }


# Tool 8: Re-predict after feedback.
# If no action was taken, re-prediction is skipped.
# If an intervention was simulated, the updated profile is passed back into
# the Random Forest to show feedback-based adaptation.
def re_predict_after_feedback_tool(state: StudentAgentState, model) -> StudentAgentState:
    intervention = state["selected_intervention"]

    if intervention == "No action needed - continue monitoring":
        # Since no intervention was applied, no re-prediction is needed.
        trace = state.get("execution_trace", [])
        trace.append("Tool skipped: no re-prediction needed because no action was taken")

        return {
            **state,
            "updated_prediction": state["prediction"],
            "updated_confidence": state["confidence"],
            "execution_trace": trace
        }

    updated_student = pd.Series(state["updated_student"])

    X_student = pd.DataFrame(
        [{feature: float(updated_student[feature]) for feature in FEATURES}],
        columns=FEATURES
    )

    probabilities = model.predict_proba(X_student)[0]
    classes = model.classes_

    prob_dict = {c: float(p) for c, p in zip(classes, probabilities)}
    prediction = max(prob_dict, key=prob_dict.get)
    confidence = prob_dict[prediction]

    trace = state.get("execution_trace", [])
    trace.append("Tool used: re-prediction after feedback")

    return {
        **state,
        "updated_prediction": prediction,
        "updated_confidence": confidence,
        "execution_trace": trace
    }



# =========================================================
# 5. Batch triage across all students
# =========================================================
# The one-student demo is useful for explaining the workflow step by step.
# This batch triage section demonstrates generalization and autonomy at scale:
# the same prediction, explanation, recommendation, and safety logic is applied
# to all 1000 simulated students to create a staff-facing priority queue.

# Batch helper: Convert safety results into staff-facing queue labels.
# This supports the report's claim that the system can prioritize limited
# student-support staff capacity.
def priority_label(prediction: str, confidence: float, safety_status: str) -> str:
    """
    Converts the agent decision into a staff-facing priority queue label.
    This helps staff decide what to review first when the agent scans all students.
    """

    if safety_status == "No action needed":
        return "Monitor only"

    if safety_status == "Human review required" and prediction == "High" and confidence >= 0.75:
        return "Urgent human review"

    if safety_status == "Human review required":
        return "Human review"

    if safety_status == "Safe to automate":
        return "Support recommended"

    return "Review"


# Batch helper: Sort the staff queue by urgency.
def priority_order(priority: str) -> int:
    """
    Lower number means higher priority in the staff queue.
    """

    order = {
        "Urgent human review": 1,
        "Human review": 2,
        "Support recommended": 3,
        "Monitor only": 4,
        "Review": 5,
    }

    return order.get(priority, 5)


@st.cache_data(show_spinner=False)
# Batch tool: Run triage across the full simulated student population.
# This does not apply interventions. It creates a prioritization view only.
def run_batch_triage(_model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the agent's prediction, explanation, recommendation, and safety logic
    across every student in the simulated dataset.

    This does not simulate interventions and does not update the student records.
    It is used to create a staff-facing triage queue.
    """

    X_all = df[FEATURES].apply(pd.to_numeric, errors="coerce").astype(float)
    all_probabilities = _model.predict_proba(X_all)
    classes = list(_model.classes_)

    triage_rows = []

    for row_index, (_, student_row) in enumerate(df.iterrows()):
        probabilities = all_probabilities[row_index]
        prob_dict = {c: float(p) for c, p in zip(classes, probabilities)}
        prediction = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[prediction]

        state: StudentAgentState = {
            "student": student_row.to_dict(),
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": prob_dict,
            "execution_trace": []
        }

        state = explain_risk_tool(state)
        state = recommend_intervention_tool(state)
        state = safety_checker_tool(state)

        priority = priority_label(
            prediction=state["prediction"],
            confidence=state["confidence"],
            safety_status=state["safety_status"]
        )

        triage_rows.append({
            "student_id": student_row["student_id"],
            "predicted_risk": state["prediction"],
            "confidence": round(float(state["confidence"]), 3),
            "priority": priority,
            "safety_status": state["safety_status"],
            "recommended_action": state["selected_intervention"],
            "top_risk_factors": ", ".join(state["risk_factors"][:4]),
            "priority_order": priority_order(priority)
        })

    triage_df = pd.DataFrame(triage_rows)

    triage_df = triage_df.sort_values(
        by=["priority_order", "confidence"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return triage_df


# =========================================================
# 6. Build LangGraph agent
# =========================================================
# LangGraph coordinates the tool sequence.
# The graph represents the agentic plan used in the report

# Build the LangGraph workflow.
# Each graph node corresponds to one modular tool above.
def build_langgraph_agent(model):
    if StateGraph is None:
        raise ImportError("langgraph is not installed. Please run: pip install -r requirements.txt")

    graph = StateGraph(StudentAgentState)

    graph.add_node("predict_risk", lambda state: predict_risk_tool(state, model))
    graph.add_node("explain_risk", explain_risk_tool)
    graph.add_node("recommend_intervention", recommend_intervention_tool)
    graph.add_node("safety_check", safety_checker_tool)
    graph.add_node("generate_message", generate_message_tool)
    graph.add_node("simulate_feedback", simulate_feedback_tool)
    graph.add_node("re_predict_after_feedback", lambda state: re_predict_after_feedback_tool(state, model))

    graph.set_entry_point("predict_risk")
    graph.add_edge("predict_risk", "explain_risk")
    graph.add_edge("explain_risk", "recommend_intervention")
    graph.add_edge("recommend_intervention", "safety_check")
    graph.add_edge("safety_check", "generate_message")
    graph.add_edge("generate_message", "simulate_feedback")
    graph.add_edge("simulate_feedback", "re_predict_after_feedback")
    graph.add_edge("re_predict_after_feedback", END)

    return graph.compile()


# =========================================================
# 7. Streamlit UI
# =========================================================
# The interface is organized around the same story used in the report and demo:
# data/model, one-student agent workflow, architecture trace, commercial stress
# test, and all-students batch triage.

st.set_page_config(
    page_title="Student Retention Agent",
    page_icon="🎓",
    layout="wide"
)

st.title("Student Retention Agent")


with st.expander("What makes this agentic?"):
    st.write("""
This is not a simple prediction dashboard. It uses a LangGraph workflow to run multiple tools in sequence:

1. Random Forest risk prediction
2. Risk explanation
3. Context-aware intervention recommendation
4. Safety and human review check
5. Standardized support message generation
6. Feedback simulation only when action is taken
7. Re-prediction after feedback only when action is taken

This demonstrates planning, tool use, multi-step task execution, human-in-the-loop safety, and feedback-based adaptation.
""")

with st.expander("Why simulated data, and how is it research-informed?"):
    st.write("""
The student database contains 1000 simulated student records because real student records are private and require institutional approval.
However, the feature schema is informed by common open learning analytics datasets.

Relevant open-data ideas reflected in the schema:
- Academic outcomes and enrolment/retention labels, as seen in higher-education dropout datasets.
- Assessment results and virtual learning environment interactions, as seen in OULAD-style learning analytics data.
- Student support and wellbeing signals, added to reflect feedback that dropout is not only academic.

This means the dataset is simulated, but the columns are not arbitrary.
They are designed around realistic categories used in student-retention analytics.
""")

# In-memory student database.
# This persists updates while the app session is open.
if "student_db" not in st.session_state:
    st.session_state["student_db"] = generate_research_informed_student_data()

df = st.session_state["student_db"]

if StateGraph is None:
    st.error("LangGraph is not installed. Run: pip install -r requirements.txt")
    st.stop()

model, metrics, importances = train_random_forest_model(df)
agent = build_langgraph_agent(model)

# The five tabs correspond to the main evaluation story:
# 1. data/model, 2. one-student workflow, 3. traceability, 4. cost-benefit,
# and 5. autonomous batch triage at scale.
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Simulated database and model",
    "2. LangGraph Agent Demo",
    "3. Architecture Trace",
    "4. Commercial Stress Test",
    "5. All Students Triage"
])

with tab1:
    st.subheader("Research-informed simulated student database")
    st.write("""
This table acts as an in-memory student database.
When an intervention is simulated, the selected student's record is updated during the app session.
In deployment, this data layer would be replaced by secure LMS, attendance, and student information system APIs.
""")

    st.metric("Total students in database", len(df))
    st.dataframe(df, width="stretch")

    st.subheader("Feature definitions")
    feature_rows = [{"feature": k, "meaning": v} for k, v in FEATURE_EXPLANATIONS.items()]
    st.dataframe(pd.DataFrame(feature_rows), width="stretch")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
    col2.metric("Precision", f"{metrics['precision']:.2f}")
    col3.metric("Recall", f"{metrics['recall']:.2f}")
    col4.metric("F1", f"{metrics['f1']:.2f}")

    st.subheader("Random Forest feature importance")
    st.bar_chart(importances.set_index("feature"))

with tab2:
    st.subheader("Run the Student Retention Agent")

    chosen_id = st.selectbox("Choose a student profile", df["student_id"].tolist())
    student = df[df["student_id"] == chosen_id].iloc[0]

    st.write("Selected student data")
    st.dataframe(student.astype(str).to_frame("value"), width="stretch")

    if st.button("Run Agent"):
        initial_state: StudentAgentState = {
            "student": student.to_dict(),
            "execution_trace": []
        }

        result = agent.invoke(initial_state)

        # Update the in-memory database only if an intervention was actually simulated.
        # For low-risk, high-confidence cases, no action is taken, so the database remains unchanged.
        db_mask = st.session_state["student_db"]["student_id"] == chosen_id

        if result["selected_intervention"] != "No action needed - continue monitoring":
            for feature in FEATURES:
                st.session_state["student_db"].loc[db_mask, feature] = float(result["updated_student"][feature])

            st.session_state["student_db"].loc[db_mask, "risk_label"] = result["updated_prediction"]
            st.session_state["student_db"].loc[db_mask, "last_intervention"] = result["selected_intervention"]
            st.session_state["student_db"].loc[db_mask, "last_safety_status"] = result["safety_status"]
            st.session_state["student_db"].loc[db_mask, "last_model_confidence"] = round(float(result["updated_confidence"]), 3)

        st.markdown("### Step 1: Risk prediction")
        st.write(f"Predicted risk level: **{result['prediction']}**")
        st.write(f"Model confidence: **{result['confidence']:.2f}**")
        st.json(result["probabilities"])

        st.markdown("### Step 2: Explanation")
        for factor in result["risk_factors"]:
            st.write(f"- {factor}")

        st.markdown("### Step 3: Context-aware intervention ranking")
        st.dataframe(pd.DataFrame(result["recommendations"]), width="stretch")

        st.markdown("### Step 4: Safety and human-in-the-loop check")
        st.write(f"Decision: **{result['safety_status']}**")
        st.write(result["safety_reason"])

        st.markdown("### Step 5: Standardized supportive message")
        st.info(result["message_status"])
        st.text_area("Generated message", result["support_message"], height=240)

        st.markdown("### Step 6: Feedback-based adaptation")

        if result["selected_intervention"] == "No action needed - continue monitoring":
            st.info(
                "No intervention was simulated because this is a Low-risk, high-confidence case. "
                "The student record was not updated; the system will continue monitoring future data."
            )

            st.markdown("### Current database record")
            current_db_row = st.session_state["student_db"][
                st.session_state["student_db"]["student_id"] == chosen_id
            ]
            st.dataframe(current_db_row.astype(str), width="stretch")

        else:
            col_before, col_after = st.columns(2)

            with col_before:
                st.write("Before intervention")
                st.write(f"Risk: **{result['prediction']}**")
                st.dataframe(pd.Series(result["student"])[FEATURES].to_frame("before"))

            with col_after:
                st.write("After simulated intervention")
                st.write(f"Updated risk: **{result['updated_prediction']}**")
                st.write(f"Updated confidence: **{result['updated_confidence']:.2f}**")
                st.dataframe(pd.Series(result["updated_student"])[FEATURES].to_frame("after"))

            st.success("The in-memory student database has been updated with the simulated post-intervention values.")

            st.markdown("### Updated database record")
            updated_db_row = st.session_state["student_db"][
                st.session_state["student_db"]["student_id"] == chosen_id
            ]
            st.dataframe(updated_db_row.astype(str), width="stretch")

        st.session_state["last_result"] = result

with tab3:
    st.subheader("LangGraph execution trace")

    st.write("""
This trace shows the agentic workflow and tool-use sequence.
It is useful for demonstrating that the system is not a simple prediction dashboard.
""")

    result = st.session_state.get("last_result")

    if not result:
        st.warning("Run the agent in Tab 2 first.")
    else:
        for i, step in enumerate(result["execution_trace"], start=1):
            st.write(f"{i}. {step}")

        st.subheader("Architecture")
        st.code("""
LangGraph StateGraph:
predict_risk
explain_risk
recommend_intervention
safety_check
generate_message
simulate_feedback
re_predict_after_feedback
""")

        st.subheader("Communication policy")
        st.code("""
No LLM is used for student-facing communication.

Reason:
- LLMs can hallucinate or produce inconsistent wording.
- Student support communication should be safe, auditable, and consistent.

Current system:
- generates a standardized support message
- uses anonymized student IDs
- avoids the label "at-risk"
- routes sensitive, high-risk, or low-confidence cases to human review
- generates no student-facing message for Low-risk, high-confidence cases

Future deployment:
- approved messages could be sent through a university email or notification API
""")

with tab4:
    # This tab supports the report's profit logic and commercial stress test.
    # It is a simple scenario calculator rather than proof of causal retention impact.
    st.subheader("Commercial stress test")

    st.write("""
This tab estimates whether the system could generate value by protecting tuition revenue.
""")

    tuition_per_student = st.number_input("Estimated annual tuition revenue per retained student", value=35000, step=1000)
    students_retained = st.number_input("Number of additional students retained", value=5, step=1)
    monthly_platform_cost = st.number_input("Estimated monthly AI/platform operating cost", value=1500, step=500)
    months = st.number_input("Months of operation", value=12, step=1)

    saved_revenue = tuition_per_student * students_retained
    operating_cost = monthly_platform_cost * months
    net_value = saved_revenue - operating_cost

    col1, col2, col3 = st.columns(3)
    col1.metric("Revenue protected", f"${saved_revenue:,.0f}")
    col2.metric("Operating cost", f"${operating_cost:,.0f}")
    col3.metric("Estimated net value", f"${net_value:,.0f}")

    st.write("""
The business argument is that retaining even a small number of students can protect more tuition revenue
than the system costs to run.
""")


with tab5:
    # This tab demonstrates the scalable version of the agent.
    # It applies the same workflow logic across all simulated students in one scan.
    st.subheader("All Students Triage")

    st.write("""
This tab shows the more autonomous version of the agent workflow.
Instead of checking only one selected student, the system scans all 1000 simulated student records
and creates a staff-facing support queue.
""")

    st.info(
        "This batch scan does not simulate interventions and does not update student records. "
        "It is used to prioritize attention across the full student dataset."
    )

    if st.button("Run Batch Triage Across All Students"):
        with st.spinner("Scanning all student profiles and creating support queue..."):
            triage_df = run_batch_triage(model, st.session_state["student_db"])

        st.session_state["triage_df"] = triage_df

    triage_df = st.session_state.get("triage_df")

    if triage_df is None:
        st.warning("Click the button above to run the batch triage scan.")
    else:
        total_students = len(triage_df)
        urgent_count = int((triage_df["priority"] == "Urgent human review").sum())
        human_review_count = int((triage_df["safety_status"] == "Human review required").sum())
        safe_to_automate_count = int((triage_df["safety_status"] == "Safe to automate").sum())
        monitor_count = int((triage_df["safety_status"] == "No action needed").sum())

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Students scanned", total_students)
        col2.metric("Urgent review", urgent_count)
        col3.metric("Human review required", human_review_count)
        col4.metric("Safe to automate", safe_to_automate_count)
        col5.metric("Monitor only", monitor_count)

        st.markdown("### Prioritized support queue")
        st.write("""
The queue is sorted so that urgent and human-review cases appear first.
This helps staff focus on the students who need attention most.
""")

        display_cols = [
            "student_id",
            "predicted_risk",
            "confidence",
            "priority",
            "safety_status",
            "recommended_action",
            "top_risk_factors",
        ]

        st.dataframe(triage_df[display_cols], width="stretch")

        st.markdown("### Triage summary by priority")
        priority_summary = (
            triage_df.groupby("priority")
            .size()
            .reset_index(name="student_count")
            .sort_values("student_count", ascending=False)
        )
        st.dataframe(priority_summary, width="stretch")

        st.markdown("### Triage summary by predicted risk")
        risk_summary = (
            triage_df.groupby("predicted_risk")
            .size()
            .reset_index(name="student_count")
            .sort_values("student_count", ascending=False)
        )
        st.dataframe(risk_summary, width="stretch")

        st.markdown("### Top 20 cases for staff review")
        review_cases = triage_df[
            triage_df["safety_status"] == "Human review required"
        ].head(20)

        if review_cases.empty:
            st.success("No human-review cases found in the current scan.")
        else:
            st.dataframe(review_cases[display_cols], width="stretch")

        st.markdown("### How to interpret this tab")
        st.write("""
For the live demo, one selected student is shown in detail so the workflow is easy to inspect.
This tab shows the scalable version: the same logic can be applied across all students in one scan.
In deployment, this could run weekly after updated LMS, assessment, attendance, and support data are available.
""")

