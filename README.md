# Student Retention Agent

## Project summary

This repository contains the Streamlit proof-of-concept for **Student Retention Agent**, the system described in the final report, *From Prediction to Support: A Human-in-the-Loop Student Retention Agent*.

The prototype demonstrates a bounded, human-in-the-loop agentic workflow for student retention support. It does not use real student records. Instead, it generates a research-informed simulated dataset of 1000 student profiles so the workflow can be tested safely and reproduced without exposing private data.

The system is designed around a specific gap in student-success analytics: prediction alone is not enough. Once a student risk signal exists, staff still need to understand why the case was flagged, decide what kind of support is suitable, avoid over-intervention, and route sensitive cases to human review.

## What the app demonstrates

The app includes five tabs:

1. **Simulated database and model**  
   Shows the 1000-student simulated dataset, feature definitions, Random Forest metrics, and feature importance.

2. **LangGraph Agent Demo**  
   Runs the full workflow for one selected student so the reasoning is easy to inspect step by step.

3. **Architecture Trace**  
   Displays the LangGraph execution trace and communication policy.

4. **Commercial Stress Test**  
   Estimates value protected by retaining students compared with estimated operating cost.

5. **All Students Triage**  
   Scans all 1000 simulated student records in one batch and creates a prioritized support queue for staff.

## Agentic workflow

The project is not a simple chat interface or static dashboard. It uses a single tool-using agent coordinated with LangGraph. The workflow is:

1. Observe a structured student profile.
2. Predict Low, Medium, or High retention risk using a supervised Random Forest classifier.
3. Explain the main risk factors using interpretable thresholds.
4. Generate personalized support recommendations from the full risk profile.
5. Apply a safety and human-review policy.
6. Generate a standardized student-support message only when appropriate.
7. Simulate feedback after an intervention only when action is taken.
8. Re-predict risk after the simulated update only when action is taken.
9. Scan all students in batch mode to create a staff-facing priority queue.

This supports the report's agentic autonomy argument because the system performs multi-step task execution, uses tools, maintains state, withholds unnecessary action, and scales from one inspected case to all simulated students.

## Data

The dataset is generated inside `app.py` by the function:

```bash
generate_research_informed_student_data(n=1000, seed=42)
```

The data are simulated because real student records are private and would require institutional approval, ethical review, secure storage, access control, and data governance.

The schema is inspired by open learning analytics datasets and common student-retention indicators:

| Feature | Meaning |
| --- | --- |
| `student_id` | Anonymous simulated identifier |
| `gpa` | Academic performance score on a 0 to 9 scale |
| `attendance_rate` | Simulated attendance or participation percentage |
| `lms_logins_week` | Weekly LMS engagement count |
| `assignment_completion` | Percentage of submitted or completed coursework |
| `missed_assignments` | Count of missed or incomplete assignments |
| `grade_trend` | Recent performance change |
| `survey_wellbeing` | Wellbeing or support proxy on a 1 to 10 scale |
| `support_requests` | Count of support interactions or support-need signals |
| `risk_label` | Generated Low, Medium, or High target label |

The generated labels are based on a hidden risk score and divided into Low, Medium, and High categories using quantiles so all classes are represented.

## Model

The app trains a supervised `RandomForestClassifier` from scikit-learn.

Current setup:

- Dataset size: 1000 simulated records
- Split: 75% training and 25% testing
- Estimators: 300
- Maximum depth: 9
- Minimum samples per leaf: 3
- Class weighting: balanced
- Random seed: 7

The app displays accuracy, weighted precision, weighted recall, and weighted F1. With the current seed and configuration, the displayed rounded results are approximately:

- Accuracy: 0.73
- Precision: 0.74
- Recall: 0.73
- F1: 0.74

These metrics are useful for validating that the prototype runs consistently, but they should not be interpreted as real-world retention prediction performance because the data are simulated.

## Safety and human review

The system includes explicit action-control rules:

| Case | Decision |
| --- | --- |
| Low risk and confidence at least 0.60 | No action, monitor only |
| Confidence below 0.60 | Human review |
| High-risk prediction | Human review |
| Advisor check-in | Human review |
| Wellbeing support referral | Human review |
| Financial or support services check-in | Human review |
| Low or Medium risk with low-impact support | Safe to automate |

Low-risk, high-confidence students receive:

- no message
- no intervention simulation
- no database update

This is important because responsible autonomy includes knowing when not to act.

## Communication policy

The prototype does **not** use an LLM for student-facing messages. Messages are standardized and deliberately avoid internal risk labels such as "at-risk". This supports safety, consistency, and auditability.

A future version could use Agentic RAG to retrieve university policies or support-service guidelines before recommending an intervention, but that is not required for the current prototype to run.

## Commercial stress test

The commercial logic is that a university or student-success office could use the system as:

- a SaaS retention-support module
- an internal decision-support layer
- an integration with LMS, SIS, or advising platforms

The current prototype has no LLM token cost because the prediction model runs locally and messages are standardized. In production, the main costs would likely be secure hosting, integration with institutional systems, monitoring, governance, and staff workflow support.

The commercial stress test tab estimates:

```text
protected tuition revenue - operating cost = estimated net value
```

This is a scenario analysis only. It is not a causal proof that the system retains students.

## Trust, robustness, and limitations

Important limitations:

- The dataset is simulated.
- The intervention effects are simulated.
- The system has not been validated on real institutional data.
- Fairness cannot be fully evaluated without governed real data.
- The system does not try to prevent all student departures.
- Some students leave for valid reasons such as transfer, planned leave, personal choice, or career change.
- Sensitive cases remain under human review.

Edge-case handling in the current prototype includes:

- low-confidence predictions routed to human review
- high-risk cases routed to human review
- sensitive intervention types routed to human review
- low-risk high-confidence cases monitored only
- no LLM-generated student-facing messages

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

If `streamlit` is not available as a command, use:

```bash
python -m streamlit run app.py
```

## Suggested demo flow

For a clear roadshow demonstration:

1. Open **Simulated database and model** to show the data and metrics.
2. Open **LangGraph Agent Demo** and run one selected student profile.
3. Explain the prediction, explanation, recommendation, safety check, message/no-message decision, and feedback update.
4. Open **Architecture Trace** to show tool sequence.
5. Open **All Students Triage** and run the batch scan across all 1000 students.
6. Open **Commercial Stress Test** to explain the cost-benefit logic.

## File structure

```text
app.py
requirements.txt
README.md
```

## Code and data availability

The submitted code contains the full prototype. The simulated data are generated programmatically by `app.py`, so no private student data are included or released.

## Academic integrity note

This app is a functional proof-of-concept using simulated data and simulated intervention effects. It should be presented as a controlled prototype, not as a production-ready retention system or as proof of real-world causal impact.
