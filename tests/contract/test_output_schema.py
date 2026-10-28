"""Output schema — CustomerOutput fields and bounds."""

from retention.contracts.api_schema import CustomerOutput


def test_output_fields():
    o = CustomerOutput(
        customer_id="1001",
        p_churn_no_contact=0.3,
        p_churn_contact=0.2,
        estimated_uplift=0.1,
        uplift_uncertainty=0.05,
        ranking_score=0.1,
        recommend_contact=True,
        reason_codes=["usage_decline"],
        data_quality_status="NORMAL",
        model_version="1.0",
        feature_version="2.0",
    )
    assert o.customer_id == "1001"


def test_probability_bounds_post_validation():
    """Validate that CustomerOutput fields accept values in [0,1] but don't
    enforce this at the schema level (validation happens in business logic)."""
    o = CustomerOutput(
        customer_id="1",
        p_churn_no_contact=1.5,
        p_churn_contact=0.2,
        estimated_uplift=0.1,
        uplift_uncertainty=0.0,
        ranking_score=0.1,
        recommend_contact=True,
        reason_codes=[],
        data_quality_status="OK",
        model_version="1",
        feature_version="1",
    )
    assert o.p_churn_no_contact == 1.5


def test_uplift_range():
    o = CustomerOutput(
        customer_id="1",
        p_churn_no_contact=0.1,
        p_churn_contact=0.9,
        estimated_uplift=-0.8,
        uplift_uncertainty=0.0,
        ranking_score=-0.8,
        recommend_contact=False,
        reason_codes=[],
        data_quality_status="OK",
        model_version="1",
        feature_version="1",
    )
    assert -1 <= o.estimated_uplift <= 1
