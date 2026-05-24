import os

import pandas as pd
from pandas import DataFrame
from ragas import Dataset

from unlp_2026_submission.entities import QuestionWithContext


def _format_dataframe(df: DataFrame) -> DataFrame:
    df.columns = [c.strip() for c in df.columns]

    answer_cols = ["A", "B", "C", "D", "E", "F"]

    required = {
        "Question_ID",
        "Domain", "N_Pages",
        "Question",
        "Correct_Answer",
        "Doc_ID",
        "Page_Num",
        "Page_Text",
        *answer_cols
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    questions: list[QuestionWithContext] = []

    for i, row in df.iterrows():
        answers_raw = [
            row[c].strip() if pd.notna(row[c]) else ""
            for c in answer_cols
        ]
        answers = [a for a in answers_raw if a != ""]
        correct_letter = row["Correct_Answer"].strip().upper()

        if correct_letter not in answer_cols:
            raise ValueError(
                f"Row {i}: Correct_Answer must be one of {answer_cols},"
                f"got: {correct_letter!r}"
            )

        questions.append(
            QuestionWithContext(
                question_id=row["Question_ID"],
                question_text=row["Question"].strip(),
                answers=answers,
                correct_answer=correct_letter,
                domain=row["Domain"].strip(),
                n_pages=row["N_Pages"],
                doc_id=row["Doc_ID"],
                page_num=row["Page_Num"],
                doc_text=row["Page_Text"].strip()
            )
        )

    return pd.DataFrame(questions)

def get_faithfulness_full_dataset(
        data_root_dir: str
):
    df = pd.read_csv(os.path.join(data_root_dir, "dev_questions_with_context.csv"))

    formatted_df = _format_dataframe(df)

    dataset = Dataset.from_pandas(
        dataframe=formatted_df,
        name="full_qa_with_context",
        backend="local/csv",
        root_dir=''
    )

    return dataset


def get_faithfulness_sports_dataset(
        data_root_dir: str
):
    df = pd.read_csv(os.path.join(data_root_dir, "dev_questions_with_context.csv"))

    sports_domain_df = df[df["Domain"].str.strip() == "sport"]

    formatted_df = _format_dataframe(sports_domain_df)

    dataset = Dataset.from_pandas(
        dataframe=formatted_df,
        name="sports_qa_with_context",
        backend="local/csv",
        root_dir=''
    )

    return dataset

def get_faithfulness_medicine_dataset(
        data_root_dir: str
):
    df = pd.read_csv(os.path.join(data_root_dir, "dev_questions_with_context.csv"))

    sports_domain_df = df[df["Domain"].str.strip() == "medicine"]

    formatted_df = _format_dataframe(sports_domain_df)

    dataset = Dataset.from_pandas(
        dataframe=formatted_df,
        name="medicine_qa_with_context",
        backend="local/csv",
        root_dir=''
    )

    return dataset

