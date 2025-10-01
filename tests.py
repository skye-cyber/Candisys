import joblib
import numpy as np
from config import MODEL_DIR


class CandidateTester:
    def __init__(self, model_path="decision_tree_model.pkl"):
        # Load trained model
        self.model = joblib.load(model_path)

    def make_features(self, applicant, employer):
        """
        Construct relative features for model input
        and also return them for inspection.
        """
        diffs = {
            "age_diff": applicant["age"] - employer["age"],
            "education_diff": applicant["education"] - employer["education"],
            "experience_diff": applicant["experience"] - employer["experience"],
            "tech_score_diff": applicant["technical_score"]
            - employer["technical_score"],
            "interview_score_diff": applicant["interview_score"]
            - employer["interview_score"],
            "prev_employment_match": int(
                applicant["prev_employment"] == employer["prev_employment"]
            ),
        }
        return diffs, np.array([list(diffs.values())])

    def evaluate(self, applicant, employer):
        diffs, features = self.make_features(applicant, employer)

        # Predict probabilities
        prob = self.model.predict_proba(features)[0]
        pred = self.model.predict(features)[0]

        # Format results
        print("\n--- Candidate Evaluation ---")
        print("Applicant vs Employer Differences (Applicant - Employer):")
        for k, v in diffs.items():
            print(f"  {k}: {v}")

        print(
            "\nPrediction:",
            "\033[1;92mSuitable ✅\033[0m" if pred else "\033[31mNot Suitable\033[0m",
        )
        print(
            f"Confidence -> Suitable: {prob[1]*100:.2f}%, Not Suitable: {prob[0]*100:.2f}%"
        )
        print("---------------------------\n")


if __name__ == "__main__":
    # Example Employer Requirement
    employer = {
        "age": 30,
        "education": 3,  # 1=Highschool, 2=Bachelor, 3=Masters
        "experience": 5,
        "technical_score": 80,
        "interview_score": 75,
        "prev_employment": True,
    }

    # Example Applicant
    applicant = {
        "age": 27,
        "education": 3,
        "experience": 6,
        "technical_score": 85,
        "interview_score": 90,
        "prev_employment": True,
    }

    tester = CandidateTester(MODEL_DIR / "RFClassifier.pkl")
    tester.evaluate(applicant, employer)
