from datetime import date
from typing import Any, Dict, List, Literal, Optional, Union, get_args, get_origin
from enum import Enum
import types
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


def typeinfo(annotation: Any) -> str:
    """Render a readable type name from a Pydantic FieldInfo annotation."""

    if annotation is None or annotation is type(None):  # noqa: E721
        return "null"
    if annotation is Any:
        return "any"

    # Primitive normalization (match requested output)
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    if annotation is bool:
        return "bool"
    if annotation is str:
        return "str"

    origin = get_origin(annotation)

    # `Annotated[T, ...]` -> unwrap to `T`
    if origin is getattr(types, "AnnotatedAlias", object()) or str(origin) == "typing.Annotated":
        args = get_args(annotation)
        return typeinfo(args[0]) if args else "any"
    if origin is getattr(__import__("typing"), "Annotated", object()):
        args = get_args(annotation)
        return typeinfo(args[0]) if args else "any"

    # `T | U` (py3.10+) and `Union[T, U]`
    if origin is Union or isinstance(annotation, types.UnionType):
        args = list(get_args(annotation))
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if len(non_none) != len(args):
            if len(non_none) == 1:
                return f"{typeinfo(non_none[0])}|optional"
            return "|".join(typeinfo(a) for a in non_none) + "|optional"
        return "|".join(typeinfo(a) for a in args)

    if origin in (list, List):
        (item_t,) = get_args(annotation) or (Any,)
        return f"list[{typeinfo(item_t)}]"

    if isinstance(annotation, type):
        # Normalize common typing-ish names while keeping unknowns readable
        return getattr(annotation, "__name__", str(annotation))

    # Fallback for uncommon typing constructs
    return str(annotation).replace("typing.", "")


class ImpactLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    TRANSFORMATIVE = "transformative"


# ────────────────────────────────────────────────
# Base class — updated version
# ────────────────────────────────────────────────
class NewsSummaryBase(BaseModel):
    """Base structure shared by all domain-specific news summaries"""

   
    headline: str = Field(
        ..., description="Exact or slightly cleaned headline of the article"
    )
    tldr: str = Field(
        ...,
        description="One crisp sentence that captures the core message / main event of the article",
    )
    key_facts: List[str] = Field(
        ...,
        min_items=3,
        max_items=8,
        description="3–8 short, factual bullet-point statements that convey the most important who/what/when/where/how details",
    )
    companies: List[str] = Field(
        default_factory=list,
        description="Names of the main companies explicitly mentioned or central to the story",
    )
    event_type: Optional[str] = Field(
        None,
        description=(
            "Short string (lowercase-with-underscores preferred) describing the primary type of event or news angle. "
            "Use consistent phrasing within a domain. Examples by domain:\n"
            "• AI: 'model_release', 'agent_launch', 'enterprise_adoption_case', 'safety_regulation_update', 'multimodal_breakthrough'\n"
            "• Cyber: 'ransomware_attack', 'zero_day_disclosure', 'supply_chain_breach', 'ai_enhanced_exploit', 'state_sponsored_campaign'\n"
            "• Hardware/HPC: 'chip_launch', 'platform_announcement', 'supply_chain_disruption', 'sovereign_funding_round', 'foundry_partnership'\n"
            "• Robotics/AV/Drones: 'humanoid_demo', 'warehouse_deployment', 'av_funding_round', 'drone_swarm_test', 'regulation_change'\n"
            "• Startup/Corp: 'series_a', 'acquisition_announced', 'merger_completed', 'strategic_partnership', 'ipo_filing'\n"
            "• Financial/Markets: 'earnings_beat', 'stock_reaction', 'analyst_upgrade', 'sector_rotation', 'sec_filing_update'\n"
            "• Logistics/Aviation: 'route_disruption', 'freight_rate_spike', 'aircraft_order', 'supply_chain_bottleneck', 'cyber_incident_on_cargo'\n"
            "• Macro/Economy: 'oil_price_shock', 'gdp_forecast_revision', 'inflation_spike', 'rate_cut_signal', 'commodity_demand_shift'\n"
            "Leave blank if no clear single event type dominates."
        ),
    )
    cross_domain_significance: List[str] = Field(
        default_factory=dict,
        description=(
            "Short sentences explaining why this story matters to other tracked domains. "
            "Keys should be lowercase domain labels (ai, cybersecurity, robotics, logistics, aviation, hardware, macro, markets, startups). "
            "Values are 1-sentence implications."
        ),
    )
    geopolitical_context: Optional[str] = Field(
        None,
        description=(
            "Short kebab-case label (lowercase, hyphens) for the main geopolitical, trade, or macro driver mentioned or implied in the article. "
            "If no significant geopolitical/macro element is present, leave as null.\n"
            "Current frequent examples in 2026:\n"
            "• iran_conflict\n"
            "• us_china_tensions\n"
            "• red_sea_disruption\n"
            "• tariff_volatility\n"
            "• taiwan_strait\n"
            "• rare_earth_controls\n"
            "For future or emerging situations, create a concise new label (3–6 words max, kebab-case). "
            "Examples of potential future labels: 'arctic_shipping_rivalry', 'lithium_supply_ban', 'africa_mineral_conflict', 'cyber_arms_race_escalation'.\n"
            "Aim for consistency and searchability — avoid full sentences or overly long phrases."
        ),
    )
    impact_level: Optional[str] = Field(
        None,
        description="Assessed severity / importance of the news item for the primary domain and broader ecosystem. Allowed: null, low, medium, high, critial, transformative",
    )
    future_outlook: Optional[str] = Field(
        None,
        description="One forward-looking sentence describing expected implications or trajectory between now and 2030",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Free-form keywords / themes that help filtering and clustering (e.g. 'agentic', 'sovereign_compute', 'defense_tech')",
    )

    @classmethod
    def schema(cls):
        return "\n".join(
            f"{fname}: {typeinfo(finfo.annotation)}|{finfo.description}"
            for fname, finfo in cls.model_fields.items()
        )

    def __str__(self):
        return self.model_dump_json()


# ────────────────────────────────────────────────
# Domain-specific models (inherit from base)
# ────────────────────────────────────────────────


class AINewsSummary(NewsSummaryBase):
    """Summary focused on AI models, agents, enterprise adoption, breakthroughs"""

    models_or_agents: List[str] = Field(
        default_factory=list,
        description="Names or codenames of AI models, agents or frameworks mentioned (e.g. 'Grok-4', 'o3-mini', 'Perplexity PC agent')",
    )
    researchers: List[str] = Field(
        default_factory=list,
        description="Names of key researchers, authors or quoted experts",
    )
    benchmark_scores: Dict[str, str] = Field(
        default_factory=dict,
        description="Reported performance numbers on standard benchmarks (key: benchmark name, value: score)",
    )
    claimed_productivity_lift_pct: Optional[float] = Field(
        None,
        description="Percentage productivity / efficiency gain claimed for users or enterprises",
    )
    enterprise_adoption_rate_pct: Optional[float] = Field(
        None,
        description="Reported percentage of companies (especially Fortune 500) already using or piloting the technology",
    )
    pricing_monthly_usd: Optional[float] = Field(
        None,
        description="Monthly subscription price in USD if a commercial product is announced",
    )
    valuation_or_market_size_usd_billions: Optional[float] = Field(
        None, description="Company valuation or projected market size in billions USD"
    )


class CyberNewsSummary(NewsSummaryBase):
    """Summary focused on cybersecurity incidents, threats, vulnerabilities, defenses"""

    threat_actors: List[str] = Field(
        default_factory=list,
        description="Named or categorized attackers (e.g. 'LockBit', 'nation-state', 'unknown')",
    )
    vulnerabilities: List[str] = Field(
        default_factory=list,
        description="CVE IDs, product names or zero-day descriptions mentioned",
    )
    affected_entities: List[str] = Field(
        default_factory=list,
        description="Organizations, sectors or number of users impacted",
    )
    malware_family: Optional[str] = Field(
        None,
        description="Name of the malware family or ransomware strain if applicable",
    )
    estimated_cost_usd_millions: Optional[float] = Field(
        None,
        description="Estimated financial damage in millions USD (ransom, recovery, fines, etc.)",
    )
    records_breached: Optional[str] = Field(
        None,
        description="Number of records or users exposed (e.g. '3.2 million', 'unknown')",
    )
    attack_speed_reduction_days: Optional[int] = Field(
        None,
        description="How many days faster attacks can now be executed due to new techniques / AI",
    )
    incident_type: Literal[
        "ransomware",
        "supply_chain",
        "zero_day",
        "ai_enhanced",
        "state_sponsored",
        "shadow_ai",
        "critical_infra",
    ] = Field(..., description="Primary category of the cybersecurity event")
    business_impact_summary: str = Field(
        ...,
        description="One sentence summarizing operational, financial, reputational or regulatory consequences",
    )
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Specific defensive or mitigation steps recommended in the article",
    )
    regulatory_triggered: List[str] = Field(
        default_factory=list,
        description="Laws / regulations mentioned as being triggered or relevant (SEC, CIRCIA, GDPR, etc.)",
    )


class HardwareHPCNewsSummary(NewsSummaryBase):
    """Summary focused on chips, accelerators, compute infrastructure"""

    product_chip: str = Field(
        ...,
        description="Name or model of the chip, platform or hardware product announced",
    )
    manufacturer: str = Field(
        ..., description="Company that designs or manufactures the hardware"
    )
    use_cases: List[str] = Field(
        default_factory=list,
        description="Intended or demonstrated applications (AI training, inference, AV simulation, edge, etc.)",
    )
    performance_improvement_x: Optional[float] = Field(
        None, description="Speedup factor compared to previous generation (e.g. 2.8×)"
    )
    power_efficiency_gain_pct: Optional[float] = Field(
        None, description="Percentage improvement in performance per watt"
    )
    capex_investment_usd_billions: Optional[float] = Field(
        None,
        description="Capital expenditure announced for data centers or fabs (in billions USD)",
    )
    price_per_unit_usd: Optional[float] = Field(
        None, description="Estimated or announced price per chip/unit in USD"
    )


class RoboticsAVDronesNewsSummary(NewsSummaryBase):
    """Summary focused on robotics systems, autonomous vehicles, drones"""

    product_system: str = Field(
        ..., description="Name or model of the robot, AV, drone or system"
    )
    manufacturer: str = Field(
        ..., description="Company developing or deploying the system"
    )
    category: Literal[
        "industrial_cobot",
        "humanoid",
        "autonomous_vehicle",
        "drone_swarm",
        "warehouse_agv",
    ] = Field(..., description="Broad category of the embodied system")
    speed_or_payload_improvement: Optional[str] = Field(
        None,
        description="Key performance upgrade (e.g. '2.5 m/s top speed', '15 kg payload')",
    )
    deployment_sites_count: Optional[int] = Field(
        None,
        description="Number of real-world deployment locations or customers mentioned",
    )
    funding_raised_usd_millions: Optional[float] = Field(
        None,
        description="Funding amount raised in millions USD if funding is the main topic",
    )
    cost_per_unit_usd: Optional[float] = Field(
        None, description="Estimated cost per robot / vehicle in USD"
    )
    real_world_limitation_noted: List[str] = Field(
        default_factory=list,
        description="Practical limitations or failure modes highlighted (e.g. 'slippery floors', 'edge cases')",
    )


class StartupCorpNewsSummary(NewsSummaryBase):
    """Summary focused on startups, corporate moves, funding, M&A"""

    main_company: str = Field(..., description="Primary company the article is about")
    other_companies: List[str] = Field(
        default_factory=list,
        description="Other companies involved (acquirers, partners, competitors)",
    )
    lead_investors: List[str] = Field(
        default_factory=list, description="Names of lead or prominent investors"
    )
    acquirer: Optional[str] = Field(
        None, description="Name of the acquiring company in M&A deals"
    )
    funding_amount_usd_millions: Optional[float] = Field(
        None, description="Amount raised in the funding round (millions USD)"
    )
    round_type: Optional[str] = Field(
        None, description="Stage of funding (Seed, Series A, Late, Debt, etc.)"
    )
    pre_post_valuation_usd_billions: Optional[Dict[str, float]] = Field(
        None, description="Pre-money and/or post-money valuation in billions USD"
    )
    deal_value_usd_millions: Optional[float] = Field(
        None, description="Transaction value in M&A deals (millions USD)"
    )
    yoy_funding_growth_pct: Optional[float] = Field(
        None,
        description="Year-over-year change in funding volume for the sector or region",
    )
    strategic_rationale: str = Field(
        ...,
        description="One sentence explaining why the deal/funding/partnership makes strategic sense",
    )
    use_of_funds: Optional[str] = Field(
        None, description="Stated or inferred purpose of the capital raised"
    )


class FinancialMarketsNewsSummary(NewsSummaryBase):
    """Summary focused on stocks, earnings, filings, market movements"""

    ticker_or_index: str = Field(
        ..., description="Stock ticker(s), index or sector the article focuses on"
    )
    companies_mentioned: List[str] = Field(
        default_factory=list,
        description="Companies whose stock or performance is discussed",
    )
    stock_reaction_pct: Optional[float] = Field(
        None,
        description="Percentage change in stock price (positive or negative) after the news",
    )
    earnings_beat_miss_pct: Optional[float] = Field(
        None, description="Percentage beat or miss vs consensus on key metrics"
    )
    revenue_or_ebitda_usd_millions: Optional[float] = Field(
        None,
        description="Reported or forecasted revenue / EBITDA figure in millions USD",
    )
    forward_guidance_change_pct: Optional[float] = Field(
        None, description="Change in forward guidance (as percentage)"
    )
    valuation_multiple: Optional[str] = Field(
        None,
        description="Forward-looking valuation metric (e.g. '32x revenue', '18x EBITDA')",
    )
    financial_analysis_summary: str = Field(
        ...,
        description="One sentence summarizing the financial implication or analyst take",
    )
    sector_rotation_signal: str = Field(
        default="",
        description="Indication of money moving between sectors (e.g. 'from Mag7 to cyclicals')",
    )


class LogisticsAviationNewsSummary(NewsSummaryBase):
    """Summary focused on freight, routes, aviation orders, disruptions"""

    modes: List[Literal["air", "ocean", "truck", "multimodal"]] = Field(
        default_factory=list, description="Transport modes discussed in the article"
    )
    routes_affected: List[str] = Field(
        default_factory=list,
        description="Specific trade lanes or chokepoints impacted (Red Sea, Suez, Transpacific, etc.)",
    )
    freight_rate_change_pct: Optional[float] = Field(
        None, description="Percentage change in freight rates (positive = increase)"
    )
    order_quantity_aircraft: Optional[int] = Field(
        None, description="Number of aircraft ordered or delivered"
    )
    delay_days_average: Optional[int] = Field(
        None, description="Average delay in days for deliveries or shipments"
    )
    cost_impact_usd_millions: Optional[float] = Field(
        None,
        description="Estimated additional cost to industry / companies in millions USD",
    )
    impact_summary: str = Field(
        ..., description="One sentence summarizing operational or economic consequences"
    )
    mitigation_strategies: List[str] = Field(
        default_factory=list,
        description="Actions companies or industry are taking to respond",
    )


class MacroEconomyNewsSummary(NewsSummaryBase):
    """Summary focused on global economy, macro indicators, forecasts"""

    gdp_growth_forecast_pct: Optional[float] = Field(
        None, description="Forecasted global or regional GDP growth rate (%)"
    )
    inflation_impact_pct: Optional[float] = Field(
        None, description="Estimated change in inflation due to the event (%)"
    )
    oil_price_scenario_usd: Optional[float] = Field(
        None,
        description="Projected oil price in USD per barrel under the discussed scenario",
    )
    gold_demand_tons: Optional[float] = Field(
        None, description="Annual central-bank or investment demand for gold in tons"
    )
    macro_signal: str = Field(
        ...,
        description="One sentence summarizing the broader economic implication (recession risk, resilience, rotation, etc.)",
    )
    significance_for_markets_startups: str = Field(
        default="",
        description="How this macro development affects capital markets or startup funding environment",
    )


# ────────────────────────────────────────────────
# New shared financial core – extracted from overlapping fields
# ────────────────────────────────────────────────
class FinancialCoreMetrics(BaseModel):
    """Reusable core quantitative financial metrics common to earnings releases and SEC filings"""

    revenue_usd_millions: Optional[float] = Field(
        None, description="Total revenue reported (millions USD)"
    )
    revenue_growth_yoy_pct: Optional[float] = Field(
        None, description="Year-over-year revenue growth percentage"
    )
    net_income_usd_millions: Optional[float] = Field(
        None,
        description="Net income attributable to common shareholders (millions USD)",
    )
    eps_basic: Optional[float] = Field(
        None, description="Basic earnings per share (GAAP)"
    )
    eps_diluted: Optional[float] = Field(None, description="Diluted EPS (GAAP)")
    operating_cash_flow_usd_millions: Optional[float] = Field(
        None, description="Net cash provided by operating activities (millions USD)"
    )
    capex_usd_millions: Optional[float] = Field(
        None, description="Capital expenditures (millions USD)"
    )
    cash_equivalents_usd_millions: Optional[float] = Field(
        None,
        description="Cash, cash equivalents & short-term investments at period end",
    )
    total_debt_usd_millions: Optional[float] = Field(
        None, description="Total short + long-term debt"
    )
    key_financial_ratios: Dict[str, str] = Field(
        default_factory=dict,
        description="Important ratios (e.g. {'gross_margin_pct': '42.1', 'operating_margin_pct': '18.7', 'net_debt_to_ebitda': '1.8x'})",
    )


# ────────────────────────────────────────────────
# Derived: Earnings Report Summary
# ────────────────────────────────────────────────
class EarningsReportSummary(NewsSummaryBase):
    """Summary for earnings press releases, call transcripts, and related materials"""

    fiscal_period: str = Field(
        ..., description="Reporting period (e.g. 'Q4 2025', 'FY 2025')"
    )
    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Full company name")

    # Compose shared financial block
    financials: FinancialCoreMetrics = Field(
        ..., description="Core quantitative financial results"
    )

    # Earnings-specific extensions
    revenue_beat_miss_pct: Optional[float] = Field(
        None, description="Revenue beat/miss vs consensus (%)"
    )
    eps_beat_miss_pct: Optional[float] = Field(
        None, description="EPS beat/miss vs consensus (%)"
    )
    eps_adjusted: Optional[float] = Field(None, description="Non-GAAP / adjusted EPS")
    gross_margin_pct: Optional[float] = Field(
        None, description="Gross margin percentage"
    )
    operating_margin_pct: Optional[float] = Field(
        None, description="Operating margin percentage"
    )
    free_cash_flow_usd_millions: Optional[float] = Field(
        None, description="Free cash flow (operating CF – capex)"
    )

    next_quarter_guidance: Dict[str, str] = Field(
        default_factory=dict,
        description="Key next-quarter guidance ranges or narrative",
    )
    full_year_guidance_update: Optional[str] = Field(
        None, description="Full-year outlook change (raised/lowered/reaffirmed)"
    )
    guidance_tone: Literal["bullish", "cautious", "neutral", "mixed"] = Field("neutral")

    key_segments_performance: Dict[str, str] = Field(
        default_factory=dict, description="Segment/product/region highlights"
    )
    main_drivers: List[str] = Field(
        default_factory=list, description="Primary drivers of results"
    )
    mdna_key_takeaways: List[str] = Field(
        default_factory=list, description="Key MD&A paraphrases"
    )
    strategic_priorities: List[str] = Field(
        default_factory=list, description="Strategic / capital allocation themes"
    )
    risks_updated: List[str] = Field(
        default_factory=list, description="Updated or emphasized risks"
    )
    one_time_items: List[str] = Field(
        default_factory=list, description="Significant non-recurring items"
    )

    management_tone: Literal[
        "confident", "cautious", "defensive", "optimistic", "mixed"
    ] = Field("neutral")
    qna_hot_topics: List[str] = Field(
        default_factory=list, description="Most discussed Q&A themes"
    )
    call_sentiment_score: Optional[float] = Field(
        None, description="Optional numeric sentiment (-1.0 to +1.0)"
    )


# ────────────────────────────────────────────────
# Derived: SEC Filing Summary
# ────────────────────────────────────────────────
class SECFilingSummary(NewsSummaryBase):
    """Summary for SEC filings (10-K, 10-Q, 8-K, etc.)"""

    filing_type: Literal["10-K", "10-K/A", "10-Q", "10-Q/A", "8-K"] = Field(
        ..., description="SEC form type"
    )
    accession_number: str = Field(..., description="EDGAR accession number")
    filing_date: date = Field(..., description="Date filed with SEC")
    period_end_date: date = Field(..., description="End of reporting period")
    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Full company name as filer")

    # Compose shared financial block
    financials: FinancialCoreMetrics = Field(
        ..., description="Core quantitative financial results from statements"
    )

    # SEC-specific extensions
    mdna_key_takeaways: List[str] = Field(
        default_factory=list, description="Most material MD&A points"
    )
    material_trends_uncertainties: List[str] = Field(
        default_factory=list,
        description="Trends/uncertainties likely to impact future results",
    )
    critical_accounting_estimates: List[str] = Field(
        default_factory=list,
        description="Significant accounting judgments/sensitivities",
    )
    top_risk_factors: List[str] = Field(
        default_factory=list, description="Top / newly emphasized risk factors"
    )
    legal_proceedings_status: List[str] = Field(
        default_factory=list, description="Material litigation / contingencies summary"
    )
    business_overview_highlights: List[str] = Field(
        default_factory=list, description="Key Item 1 Business points"
    )
    strategic_initiatives: List[str] = Field(
        default_factory=list, description="Major strategic priorities / shifts"
    )
    key_exhibits_filed: List[str] = Field(
        default_factory=list,
        description="Notable exhibits (e.g. insider policy, material contracts)",
    )
    material_events_8k: Optional[str] = Field(
        None, description="For 8-K: description of triggering event"
    )


# ────────────────────────────────────────────────
# Merged model: FinancialDocumentSummary
# ────────────────────────────────────────────────
class FinancialDocumentSummary(NewsSummaryBase):
    """
    Unified model covering earnings releases, call transcripts, 10-K, 10-Q, 8-K and combinations.
    Use 'document_subtype' to distinguish primary focus.
    """

    document_subtype: Literal[
        "earnings_release",  # Press release only
        "earnings_call_transcript",  # Transcript / call summary
        "earnings_package",  # Release + call + slides
        "10-K",  # Annual report
        "10-Q",  # Quarterly report
        "8-K",  # Current report / material event
        "hybrid_earnings_filing",  # e.g. 10-Q + earnings release on same day
    ] = Field(..., description="Primary nature / most important part of the document")

    fiscal_period: str = Field(
        ..., description="Reporting period e.g. 'Q4 2025', 'FY 2025'"
    )
    ticker: str = Field(..., description="Stock ticker")
    company_name: str = Field(..., description="Company name as reported")
    period_end_date: date = Field(..., description="End date of the fiscal period")
    filing_date: Optional[date] = Field(
        None, description="SEC filing/submission date (if applicable)"
    )

    # Core financials (shared)
    financials: FinancialCoreMetrics = Field(..., description="Quantitative backbone")

    # Earnings-specific extensions (optional – often null for pure 10-K/8-K)
    revenue_beat_miss_pct: Optional[float] = Field(
        None, description="Revenue vs consensus (%)"
    )
    eps_beat_miss_pct: Optional[float] = Field(None, description="EPS vs consensus (%)")
    eps_adjusted: Optional[float] = Field(None, description="Non-GAAP/adjusted EPS")
    gross_margin_pct: Optional[float] = Field(None, description="Gross margin %")
    operating_margin_pct: Optional[float] = Field(
        None, description="Operating margin %"
    )
    free_cash_flow_usd_millions: Optional[float] = Field(
        None, description="Free cash flow"
    )

    next_quarter_guidance: Dict[str, str] = Field(
        default_factory=dict, description="Next-quarter ranges or narrative"
    )
    full_year_guidance_update: Optional[str] = Field(
        None, description="Full-year outlook change"
    )
    guidance_tone: Optional[Literal["bullish", "cautious", "neutral", "mixed"]] = Field(
        None
    )

    # Call-specific (optional – usually null for pure filings)
    management_tone: Optional[
        Literal["confident", "cautious", "defensive", "optimistic", "mixed"]
    ] = Field(None)
    qna_hot_topics: List[str] = Field(
        default_factory=list, description="Recurring or heated Q&A themes"
    )
    call_sentiment_score: Optional[float] = Field(
        None, description="Numeric sentiment if analyzed (-1.0 to +1.0)"
    )

    # MD&A / Narrative – shared but emphasized differently
    mdna_key_takeaways: List[str] = Field(
        default_factory=list, description="Most important MD&A points / explanations"
    )
    main_drivers: List[str] = Field(
        default_factory=list,
        description="Primary drivers of results (volume, price, costs, FX…)",
    )

    # Risk & Legal – more prominent in filings
    top_risk_factors: List[str] = Field(
        default_factory=list, description="Key / newly updated risks"
    )
    legal_proceedings_status: List[str] = Field(
        default_factory=list, description="Material litigation or contingencies"
    )

    # Business / Strategy – stronger in 10-K
    business_overview_highlights: List[str] = Field(
        default_factory=list, description="Core business description points"
    )
    strategic_initiatives: List[str] = Field(
        default_factory=list, description="Strategic priorities, investments, shifts"
    )

    # Other material items
    one_time_items: List[str] = Field(
        default_factory=list, description="Non-recurring charges/gains"
    )
    material_trends_uncertainties: List[str] = Field(
        default_factory=list,
        description="Trends likely to materially affect future results",
    )
    key_exhibits_filed: List[str] = Field(
        default_factory=list, description="Notable exhibits (policies, contracts…)"
    )

    # 8-K specific
    material_event_description: Optional[str] = Field(
        None, description="For 8-K: what triggered the filing"
    )


# Union type for convenience
AnyNewsSummary = Union[
    AINewsSummary,
    CyberNewsSummary,
    HardwareHPCNewsSummary,
    RoboticsAVDronesNewsSummary,
    StartupCorpNewsSummary,
    FinancialMarketsNewsSummary,
    LogisticsAviationNewsSummary,
    MacroEconomyNewsSummary,
]
