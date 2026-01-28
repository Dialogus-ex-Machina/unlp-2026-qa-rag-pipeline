"""Context Recall prompt classes and models."""

from typing import List

from pydantic import BaseModel, Field

from ragas.prompt.metrics.base_prompt import BasePrompt

class ContextRecallInput(BaseModel):
    """Input model for context recall evaluation."""

    question: str = Field(..., description="The original question asked by the user")
    context: str = Field(..., description="The retrieved context passage to evaluate")
    answer: str = Field(
        ..., description="The reference answer containing statements to classify"
    )


class ContextRecallClassification(BaseModel):
    """Classification of a single statement."""

    statement: str = Field(
        ..., description="Individual statement extracted from the answer"
    )
    reason: str = Field(
        ...,
        description="Reasoning for why the statement is or isn't attributable to context",
    )
    attributed: int = Field(
        ...,
        description="Binary classification: 1 if the statement can be attributed to context, 0 otherwise",
    )


class ContextRecallOutput(BaseModel):
    """Structured output for context recall classifications."""

    classifications: List[ContextRecallClassification] = Field(
        ..., description="List of statement classifications"
    )


class ContextRecallPrompt(BasePrompt[ContextRecallInput, ContextRecallOutput]):
    """Context recall evaluation prompt with structured input/output."""

    input_model = ContextRecallInput
    output_model = ContextRecallOutput

    instruction = """Given a context and an answer, analyze each statement in the answer and classify if the statement can be attributed to the given context or not.
Use only binary classification: 1 if the statement can be attributed to the context, 0 if it cannot.
Provide detailed reasoning for each classification."""

    examples = [
        (
            ContextRecallInput(
                question="Що ти можеш розповісти мені про Альберта Ейнштейна?",
                context="Альберт Ейнштейн (14 березня 1879 — 18 квітня 1955) — фізик-теоретик німецького походження, якого вважають одним із найвидатніших і найвпливовіших учених усіх часів. Найбільш відомий завдяки розробці теорії відносності, він також зробив важливий внесок у квантову механіку і таким чином став центральною фігурою революційної зміни наукового розуміння природи, якої досягла сучасна фізика в перші десятиліття ХХ століття. Його формула еквівалентності маси та енергії E = mc², що випливає з теорії відносності, була названа «найвідомішим рівнянням у світі». У 1921 році він отримав Нобелівську премію з фізики «за заслуги у теоретичній фізиці, і особливо за відкриття закону фотоефекту», що стало ключовим кроком у розвитку квантової теорії. Його роботи також відомі своїм впливом на філософію науки. У опитуванні 1999 року серед 130 провідних фізиків світу, проведеному британським журналом Physics World, Ейнштейн був визнаний найвидатнішим фізиком усіх часів. Його інтелектуальні досягнення та оригінальність зробили ім’я Ейнштейна синонімом геніальності.",
                answer="Альберт Ейнштейн, народився 14 березня 1879 року, був фізиком-теоретиком німецького походження, якого вважають одним із найвидатніших і найвпливовіших учених усіх часів. У 1921 році він отримав Нобелівську премію з фізики за заслуги в теоретичній фізиці. У 1905 році він опублікував 4 наукові роботи. Ейнштейн переїхав до Швейцарії у 1895 році.",
            ),
            ContextRecallOutput(
                classifications=[
                    ContextRecallClassification(
                        statement="Альберт Ейнштейн, народжений 14 березня 1879 року, фізиком-теоретиком німецького походження, якого вважають одним із найвидатніших і найвпливовіших учених усіх часів.",
                        reason="Дата народження Ейнштейна чітко згадується в контексті.",
                        attributed=1,
                    ),
                    ContextRecallClassification(
                        statement="У 1921 році він отримав Нобелівську премію з фізики за заслуги в теоретичній фізиці.",
                        reason="Точне формулювання цього твердження присутнє в наданому контексті.",
                        attributed=1,
                    ),
                    ContextRecallClassification(
                        statement="У 1905 році він опублікував 4 наукові роботи.",
                        reason="У наданому контексті немає згадок про кількість опублікованих ним робіт.",
                        attributed=0,
                    ),
                    ContextRecallClassification(
                        statement="Ейнштейн переїхав до Швейцарії у 1895 році.",
                        reason="У наданому контексті немає підтверджуючих доказів цього твердження.",
                        attributed=0,
                    ),
                ]
            ),
        ),
        (
            ContextRecallInput(
                question="Хто виграв ICC Men's T20 World Cup 2022?",
                context="Чемпіонат ICC Men's T20 World Cup 2022, який проходив з 16 жовтня по 13 листопада 2022 року в Австралії, був восьмим розіграшем цього турніру. Спочатку його було заплановано на 2020 рік, але відкладено через пандемію COVID-19. Переможцем стала збірна Англії, яка у фіналі перемогла Пакистан і здобула свій другий титул чемпіона світу ICC.",
                answer="Англія",
            ),
            ContextRecallOutput(
                classifications=[
                    ContextRecallClassification(
                        statement="Англія",
                        reason="Контекст пояснює, що Англія виграла турнір 2022 року, який спочатку був запланований на 2020 рік.",
                        attributed=1,
                    ),
                ]
            ),
        ),
        (
            ContextRecallInput(
                question="Яка найвища гора у світі?",
                context="Анди — це найдовший континентальний гірський хребет у світі, розташований у Південній Америці. Він простягається через сім країн і включає найвищі вершини Західної півкулі. Хребет відомий своїми різноманітними екосистемами, зокрема високогірним Андським плато та тропічними лісами Амазонії.",
                answer="Гора Еверест.",
            ),
            ContextRecallOutput(
                classifications=[
                    ContextRecallClassification(
                        statement="Гора Еверест.",
                        reason="Наданий контекст описує гірський хребет Анди, який не включає гору Еверест і не має прямого відношення до найвищої гори у світі.",
                        attributed=0,
                    ),
                ]
            ),
        ),
    ]
