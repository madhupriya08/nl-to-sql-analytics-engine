"""Deterministic synthetic loan-book generator.

Everything downstream -- the schema sampler, the demo answers, the
adversarial tests -- asserts against this data, so it is generated from a
fixed seed with the standard library's ``random`` module. No numpy, no
faker: fewer dependencies means the test suite runs anywhere.

The distributions are not uniform on purpose. Risk tiers, rates and
default rates correlate the way a real loan book does, so that queries
like "average interest rate by risk tier" return an ordering a human
would recognise as sane rather than noise.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from typing import Iterator

DEFAULT_ROW_COUNT = 5_000
DEFAULT_SEED = 20240817

#: Legal-entity booking the loan.
ENTITIES = ("Acme Capital", "Borealis Bank", "Cedar Financial", "Dunhill Credit")

#: Product line. Weighted so personal loans dominate, as in a real retail book.
PRODUCTS = ("Personal Loan", "Auto Loan", "Home Equity", "Credit Card", "Small Business")
PRODUCT_WEIGHTS = (0.34, 0.24, 0.14, 0.20, 0.08)

#: Credit-quality tier. The exact capitalisation matters: the schema
#: sampler surfaces these literal strings to the model so it writes
#: ``risk_tier = 'Prime'`` rather than ``'prime'``, which would silently
#: return zero rows in SQLite (its ``=`` is case-sensitive for TEXT).
RISK_TIERS = ("Prime", "Near-Prime", "Subprime")
RISK_TIER_WEIGHTS = (0.45, 0.35, 0.20)

#: (min_rate, max_rate, probability_of_default) per tier.
TIER_PROFILE = {
    "Prime": (0.049, 0.089, 0.021),
    "Near-Prime": (0.089, 0.159, 0.078),
    "Subprime": (0.159, 0.289, 0.194),
}

#: (min_amount, max_amount) per product, in whole dollars.
PRODUCT_AMOUNT_RANGE = {
    "Personal Loan": (2_000, 45_000),
    "Auto Loan": (7_500, 78_000),
    "Home Equity": (25_000, 400_000),
    "Credit Card": (500, 25_000),
    "Small Business": (15_000, 250_000),
}

ORIGINATION_START = date(2021, 1, 1)
ORIGINATION_END = date(2024, 12, 31)


@dataclass(frozen=True)
class Loan:
    """One row of the loan book.

    Field names are the column names. They are deliberately the words a
    person would use when asking a question ("loan amount", "risk tier",
    "default flag"), which makes the model's job of mapping question to
    column mostly mechanical.
    """

    loan_id: str
    entity: str
    product: str
    risk_tier: str
    loan_amount: float
    interest_rate: float
    origination_date: str  # ISO-8601 'YYYY-MM-DD'; SQLite has no date type.
    default_flag: int  # 0/1 rather than bool, so SUM()/AVG() work directly.


def _weighted_choice(rng: random.Random, options: tuple, weights: tuple) -> str:
    return rng.choices(options, weights=weights, k=1)[0]


def generate_loans(
    row_count: int = DEFAULT_ROW_COUNT, seed: int = DEFAULT_SEED
) -> Iterator[Loan]:
    """Yield ``row_count`` deterministic loans for the given ``seed``."""
    rng = random.Random(seed)
    span_days = (ORIGINATION_END - ORIGINATION_START).days

    for i in range(1, row_count + 1):
        product = _weighted_choice(rng, PRODUCTS, PRODUCT_WEIGHTS)
        risk_tier = _weighted_choice(rng, RISK_TIERS, RISK_TIER_WEIGHTS)

        rate_lo, rate_hi, default_prob = TIER_PROFILE[risk_tier]
        amount_lo, amount_hi = PRODUCT_AMOUNT_RANGE[product]

        # Higher-risk tiers skew to smaller balances, which is why the
        # amount draw is triangular rather than uniform: the same query
        # ("total exposure by tier") then shows the concentration a credit
        # analyst would expect instead of three equal bars.
        amount = rng.triangular(
            amount_lo, amount_hi, amount_lo + (amount_hi - amount_lo) * 0.35
        )

        yield Loan(
            loan_id=f"LN-{i:06d}",
            entity=rng.choice(ENTITIES),
            product=product,
            risk_tier=risk_tier,
            loan_amount=round(amount, 2),
            interest_rate=round(rng.uniform(rate_lo, rate_hi), 4),
            origination_date=(
                ORIGINATION_START + timedelta(days=rng.randint(0, span_days))
            ).isoformat(),
            default_flag=1 if rng.random() < default_prob else 0,
        )


def generate_rows(
    row_count: int = DEFAULT_ROW_COUNT, seed: int = DEFAULT_SEED
) -> Iterator[dict]:
    """Same data as :func:`generate_loans`, as plain dicts for the loader."""
    for loan in generate_loans(row_count=row_count, seed=seed):
        yield asdict(loan)
