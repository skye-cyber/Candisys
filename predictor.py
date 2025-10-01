import joblib
import numpy as np
from config import MODEL_DIR


class Predictor:
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

    def generate_feedback(self, diffs, prediction):
        """
        Analyze diffs and return reasons & recommendations.
        """
        feedback = []
        if not prediction:  # Applicant marked as Not Suitable
            if diffs["age_diff"] > 0:
                feedback.append("Applicant is older than employer’s preferred age.")
            if diffs["education_diff"] < 0:
                feedback.append("Applicant’s education level is below requirements.")
            if diffs["experience_diff"] < 0:
                feedback.append(
                    f"Needs at least {-diffs['experience_diff']} more years of experience."
                )
            if diffs["tech_score_diff"] < 0:
                feedback.append(
                    f"Improve technical score by {-diffs['tech_score_diff']} points."
                )
            if diffs["interview_score_diff"] < 0:
                feedback.append(
                    f"Perform better in interviews (+{-diffs['interview_score_diff']} needed)."
                )
            if diffs["prev_employment_match"] == 0:
                feedback.append(
                    "Previous employment history does not meet employer’s requirement."
                )

        return feedback if feedback else ["No issues detected."]

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

        # Add feedback
        print("\nFeedback:")
        for f in self.generate_feedback(diffs, pred):
            print(f" - {f}")

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
        "technical_score": 70,
        "interview_score": 60,
        "prev_employment": False,
    }

    tester = Predictor(MODEL_DIR / "DecisionTree.pkl")
    tester.evaluate(applicant, employer)
