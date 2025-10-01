import pandas as pd
from config import getLogger, original_dataset
import numpy as np
from config import BASE_DIR
from tqdm.asyncio import tqdm

logger = getLogger()

logger.info(f"Dataset: \033[1;94m{original_dataset}\033[0m")


class DataPrep:
    def __init__(self):
        self.df = pd.read_csv(original_dataset)
        self.encoded_dt_path = BASE_DIR / "datasets/encoded_dataset.csv"
        self.open_dt_path = BASE_DIR / "datasets/open_dataset.csv"

    def run(self, sy_length=10_000, encode=False):
        """
        ap-> denotes applicant
        em-> denotes employer
        """
        # --- Step 1: Prepare applicant dataset ---
        df1 = self.filterColumns(encode=encode)
        df2 = self.synthesize(length=sy_length)
        cleaned_df = pd.concat([df1, df2], ignore_index=True).drop_duplicates()

        # --- Step 2: Prepare employer dataset ---
        employer_requirement = self.synthesize_employer_criterion(length=sy_length)

        # --- Step 3: Define column mappings ---
        applicant_map = {
            "Age": "ap_Age",
            "Education": "ap_Education",
            "YearsOfExperience": "ap_YearsOfExperience",
            "PerformanceRating": "ap_TechnicalScore",
            "InterviewScore": "ap_InterviewScore",
            "PreviouEmployment": "ap_PreviouEmployment",
        }

        employer_map = {
            "Age": "em_Age",
            "Education": "em_Education",
            "YearsOfExperience": "em_YearsOfExperience",
            "PerformanceRating": "em_TechnicalScore",
            "InterviewScore": "em_InterviewScore",
            "PreviouEmployment": "em_PreviouEmployment",
        }

        # --- Step 4: Apply mappings & enforce int where possible ---
        applicant_data = cleaned_df.rename(columns=applicant_map).astype(
            {k: "Int64" for k in applicant_map.values() if "Previou" not in k}
        )

        employer_data = employer_requirement.rename(columns=employer_map).astype(
            {k: "Int64" for k in employer_map.values() if "Previou" not in k}
        )

        # ---Step 6 Clip both to same length (keeps first min_len rows)
        # after applicant_data and employer_data are created:
        min_len = min(len(applicant_data), len(employer_data))

        ap_clipped = (
            applicant_data.reset_index(drop=True).iloc[:min_len].reset_index(drop=True)
        )
        em_clipped = (
            employer_data.reset_index(drop=True).iloc[:min_len].reset_index(drop=True)
        )

        # Step 7: obtain suitables
        suitable_bools = self.is_suitable(
            *[ap_clipped[c] for c in ap_clipped.columns],
            *[em_clipped[c] for c in em_clipped.columns],
        )

        # Convert Yes/No → Boolean (True/False)
        ap_clipped["ap_PreviouEmployment"] = ap_clipped["ap_PreviouEmployment"].map(
            {"Yes": True, "No": False}
        )
        em_clipped["em_PreviouEmployment"] = em_clipped["em_PreviouEmployment"].map(
            {"Yes": True, "No": False}
        )

        final_df = pd.concat([ap_clipped, em_clipped], axis=1)
        final_df["Suitable"] = suitable_bools  # bool dtype

        r_df = pd.DataFrame(
            {
                "age_diff": ap_clipped["ap_Age"] - em_clipped["em_Age"],
                "education_diff": ap_clipped["ap_Education"]
                - em_clipped["em_Education"],
                "experience_diff": ap_clipped["ap_YearsOfExperience"]
                - em_clipped["em_YearsOfExperience"],
                "tech_score_diff": ap_clipped["ap_TechnicalScore"]
                - em_clipped["em_TechnicalScore"],
                "interview_score_diff": ap_clipped["ap_InterviewScore"]
                - em_clipped["em_InterviewScore"],
                "prev_employment_match": ap_clipped["ap_PreviouEmployment"]
                == em_clipped["em_PreviouEmployment"],
            }
        )

        r_df["Suitable"] = suitable_bools
        # --- Step 8: Save results ---
        save_path = self.encoded_dt_path if encode else self.open_dt_path
        r_df.to_csv(save_path, index=False)

        logger.info(f"File saved to: \033[1;32m{save_path}\033[0m")

    def filterColumns(self, encode=False):
        c_age = self.df["Age"]
        """
        1-> high school
        2->Bachelor’s
        3->Master’s
        4->PhD
        """
        c_education = self.df["Education"]

        c_yearsOfExperience = self.df["TotalWorkingYears"]

        c_technicalScore = (
            self.df["PerformanceRating"] * 10
        )  # Technical test score*100 to convert to /100

        # convert to int alternatively,  Use nullable integer type#
        # s_int = s.astype("Int64")  # Capital 'I'
        c_technicalScore = c_technicalScore.astype(int)

        c_interviewScore = [np.random.randint(1, 10) for i in range(len(c_age))]  # /10

        c_previousEmployment = [
            "Yes" if previous_companies > 0 else "No"
            for previous_companies in self.df["NumCompaniesWorked"]
        ]

        self.df_data = {
            "Age": c_age,
            "Education": c_education,
            "YearsOfExperience": c_yearsOfExperience,
            "PerformanceRating": c_technicalScore,
            "InterviewScore": c_interviewScore,
            "PreviouEmployment": c_previousEmployment,
        }

        new_df = pd.DataFrame(self.df_data)

        return new_df

    def is_suitable_encoded(
        self,
        c_age,
        c_education,
        c_yearsOfExperience,
        c_technicalScore,
        c_interviewScore,
        c_previousEmployment,
    ):
        suitable = []
        for index, ap_age in enumerate(c_age):
            age_suitable = c_age[index] <= 40
            education_suitable = c_education[index] >= 2
            experience_suitable = c_yearsOfExperience[index] >= 1
            technical_score_suitable = c_technicalScore[index] >= 30
            interview_score_suitable = c_interviewScore[index] >= 5
            prev_employment_suitable = c_previousEmployment[index] == "Yes"
            suitable.append(
                "Yes"
                if all(
                    (
                        age_suitable,
                        education_suitable,
                        experience_suitable,
                        technical_score_suitable,
                        interview_score_suitable,
                        prev_employment_suitable,
                    )
                )
                else "No"
            )

        return suitable

    def synthesize_employer_criterion(self, length=10_000):
        return self.synthesize(length=length)

    def synthesize(self, length=10_000):
        max_age = 60
        min_age = 18
        min_ed = 1
        max_ed = 4
        sy_age = np.random.randint(low=min_age, high=max_age, size=length)
        sy_education = np.random.randint(low=min_ed, high=max_ed, size=length)
        sy_experience = np.random.randint(low=0, high=40, size=length)
        sy_technical_score = np.random.randint(low=1, high=100, size=length)
        sy_interview_score = np.random.randint(low=1, high=10, size=length)
        sy_prev_employment_int = np.random.randint(low=1, high=2, size=length)
        sy_prev_employment = ["Yes" if i == 1 else "No" for i in sy_prev_employment_int]

        df = pd.DataFrame(
            {
                "Age": sy_age,
                "Education": sy_education,
                "YearsOfExperience": sy_experience,
                "PerformanceRating": sy_technical_score,
                "InterviewScore": sy_interview_score,
                "PreviouEmployment": sy_prev_employment,
            }
        )

        return df

    def synthesize_suitable():
        pass

    def is_suitable(
        self,
        sy_ap_age,
        sy_ap_education,
        sy_ap_experience,
        sy_ap_technical_score,
        sy_ap_interview_score,
        sy_ap_prev_employment,
        sy_em_age,
        sy_em_education,
        sy_em_experience,
        sy_em_technical_score,
        sy_em_interview_score,
        sy_em_prev_employment,
    ):
        suitable = []

        # clip both sides to the same length
        length = min(len(sy_ap_age), len(sy_em_age))

        # convert once to lists for efficiency
        sy_ap_age = list(sy_ap_age)
        sy_ap_education = list(sy_ap_education)
        sy_ap_experience = list(sy_ap_experience)
        sy_ap_technical_score = list(sy_ap_technical_score)
        sy_ap_interview_score = list(sy_ap_interview_score)
        sy_ap_prev_employment = list(sy_ap_prev_employment)

        sy_em_age = list(sy_em_age)
        sy_em_education = list(sy_em_education)
        sy_em_experience = list(sy_em_experience)
        sy_em_technical_score = list(sy_em_technical_score)
        sy_em_interview_score = list(sy_em_interview_score)
        sy_em_prev_employment = list(sy_em_prev_employment)

        for i in tqdm(range(length), desc="Find Suitables:"):  # <-- start at 0
            sy_ap_age_suitable = sy_ap_age[i] <= sy_em_age[i]
            sy_ap_education_suitable = sy_ap_education[i] >= sy_em_education[i]
            sy_ap_experience_suitable = sy_ap_experience[i] >= sy_em_experience[i]
            sy_ap_technical_score_suitable = (
                sy_ap_technical_score[i] >= sy_em_technical_score[i]
            )
            sy_ap_interview_score_suitable = (
                sy_ap_interview_score[i] >= sy_em_interview_score[i]
            )
            sy_ap_prev_employment_suitable = (
                sy_ap_prev_employment[i] == sy_em_prev_employment[i]  # <-- fixed
            )

            suitable.append(
                all(
                    (
                        sy_ap_age_suitable,
                        sy_ap_education_suitable,
                        sy_ap_experience_suitable,
                        sy_ap_technical_score_suitable,
                        sy_ap_interview_score_suitable,
                        sy_ap_prev_employment_suitable,
                    )
                )
            )

        return suitable


if __name__ == "__main__":
    inst = DataPrep()
    inst.run(sy_length=1000_000)
