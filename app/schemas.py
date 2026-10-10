from pydantic import AliasChoices, BaseModel, Field


class CreditApplication(BaseModel):
    revolving_utilization: float = Field(
        validation_alias="RevolvingUtilizationOfUnsecuredLines"
    )
    age: float = Field(validation_alias="age")
    past_due_30_59: float = Field(
        validation_alias=AliasChoices(
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime30_59DaysPastDueNotWorse",
        )
    )
    debt_ratio: float = Field(validation_alias="DebtRatio")
    monthly_income: float | None = Field(default=None, validation_alias="MonthlyIncome")
    open_credit_lines: float = Field(
        validation_alias="NumberOfOpenCreditLinesAndLoans"
    )
    past_due_90: float = Field(validation_alias="NumberOfTimes90DaysLate")
    real_estate_lines: float = Field(
        validation_alias="NumberRealEstateLoansOrLines"
    )
    past_due_60_89: float = Field(
        validation_alias=AliasChoices(
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTime60_89DaysPastDueNotWorse",
        )
    )
    dependents: float | None = Field(default=None, validation_alias="NumberOfDependents")


class PredictionResponse(BaseModel):
    default_probability: float
    decision: str