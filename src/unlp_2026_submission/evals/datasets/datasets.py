import pandas as pd
from pandas import DataFrame
from ragas import Dataset

from unlp_2026_submission.entities import SingleAnswerQuestion


def _format_dataframe(df: DataFrame) -> DataFrame:
    df.columns = [c.strip() for c in df.columns]

    answer_cols = ["A", "B", "C", "D", "E", "F"]

    required = {"Question_ID", "Question", "Correct_Answer", *answer_cols}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    questions: list[SingleAnswerQuestion] = []

    for i, row in df.iterrows():
        answers_raw = [row[c].strip() for c in answer_cols]
        answers = [a for a in answers_raw if a != ""]
        correct_letter = row["Correct_Answer"].strip().upper()

        if correct_letter not in answer_cols:
            raise ValueError(
                f"Row {i}: Correct_Answer must be one of {answer_cols},"
                f"got: {correct_letter!r}"
            )

        questions.append(
            SingleAnswerQuestion(
                question_id=row["Question_ID"],
                question_text=row["Question"].strip(),
                answers=answers,
                correct_answer=correct_letter,
                domain=row["Domain"].strip(),
                n_pages=row["N_Pages"],
                doc_id=row["Doc_ID"],
                page_num=row["Page_Num"]
            )
        )

    return pd.DataFrame(questions)

def get_dataset():
    df = pd.read_csv("datasets/dev_questions.csv")

    formatted_df = _format_dataframe(df)

    dataset = Dataset.from_pandas(
        dataframe=formatted_df,
        name="question_answering_dev",
        backend="local/csv",
        root_dir=''
    )

    return dataset
