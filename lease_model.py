"""
Lease Modeling Dashboard

This module provides a comprehensive lease modeling tool that calculates various financial metrics
for commercial real estate leases, including NPV, payback period, and annual rent summaries.
The tool includes interactive visualizations and detailed financial reporting capabilities.

Author: Michael Morretta
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Union
import io
from datetime import datetime, timedelta
import plotly.graph_objects as go
import logging
from decimal import Decimal, ROUND_HALF_UP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MONTHS_PER_YEAR = 12
DEFAULT_DISCOUNT_RATE = 8.0
DEFAULT_ESCALATION_RATE = 3.0
MAX_LEASE_TERM_YEARS = 30
MAX_FREE_RENT_MONTHS = 24

# ---- Custom Styling (Institutional PE Aesthetic) ----
st.set_page_config(
    layout="wide",
    page_title="Lease Modeling Dashboard",
    page_icon="🏢",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/mikemorretta/CRE',
        'Report a bug': "https://github.com/mikemorretta/CRE/issues",
        'About': "# Lease Modeling Dashboard\nA tool for commercial real estate lease analysis"
    }
)

# Apply custom styling
st.markdown(
    """
    <style>
    /* Main app styling */
    .stApp {
        background-color: #1e1e1e;
        max-width: 100%;
        overflow-x: hidden;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Remove top padding and margin */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Remove the box above the title */
    .stMarkdown:first-of-type {
        display: none;
    }
    
    /* Make title more compact */
    h1 {
        margin-top: 0rem;
        margin-bottom: 0.5rem;
        padding-top: 0rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Text styling */
    html, body, [class*="css"]  {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-size: 16px;
        color: #ffffff;
        -webkit-text-size-adjust: 100%;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-rendering: optimizeLegibility;
        font-weight: 400;
        line-height: 1.5;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        font-weight: 700;
        color: #ffffff;  /* Pure white for better contrast */
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        word-wrap: break-word;
        letter-spacing: -0.5px;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Tables */
    .dataframe {
        background-color: #2d2d2d;
        border: 1px solid #3d3d3d;
        border-radius: 4px;
        width: 100%;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        margin-bottom: 0.5rem;
        font-size: 15px;
    }
    
    .dataframe th {
        background-color: #3d3d3d;
        color: #ffffff;  /* Pure white for better contrast */
        font-weight: 600;
        text-align: center;
        position: sticky;
        top: 0;
        padding: 12px 8px;
        font-size: 15px;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    .dataframe td {
        text-align: right;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-size: 15px;
        color: #ffffff;  /* Pure white for better contrast */
        padding: 12px 8px;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #2d2d2d;
        padding: 0.5rem;
        position: sticky;
        top: 0;
        height: 100vh;
        overflow-y: auto;
    }
    
    /* Form elements */
    .stNumberInput, .stTextInput {
        background-color: #2d2d2d;
        margin-bottom: 0.25rem;
        width: 100%;
    }
    
    /* Input fields - iOS specific overrides */
    @supports (-webkit-touch-callout: none) {
        /* iOS only */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {
            -webkit-text-fill-color: #000000 !important;
            color: #000000 !important;
            background-color: #ffffff !important;
            opacity: 1 !important;
        }
        
        .stTextInput > div > div > input::placeholder,
        .stNumberInput > div > div > input::placeholder {
            -webkit-text-fill-color: #666666 !important;
            color: #666666 !important;
            opacity: 1 !important;
        }
    }

    /* General input styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background-color: #ffffff !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        padding: 12px;
        font-size: 16px;
        border-radius: 8px;
        border: 1px solid #3d3d3d;
        -webkit-appearance: none;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        opacity: 1 !important;
    }

    /* Override the universal selector for input elements */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input::placeholder,
    .stNumberInput > div > div > input::placeholder {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        opacity: 1 !important;
    }

    /* Placeholder text */
    .stTextInput > div > div > input::placeholder,
    .stNumberInput > div > div > input::placeholder {
        color: #666666 !important;
        -webkit-text-fill-color: #666666 !important;
        opacity: 1 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #003366;
        color: white;
        border-radius: 8px;
        padding: 8px 12px;
        margin-top: 0.25rem;
        width: 100%;
        font-size: 16px;
        border: none;
        -webkit-appearance: none;
    }
    
    /* Mobile-specific styles */
    @media (max-width: 768px) {
        html, body, [class*="css"] {
            font-size: 17px;
            line-height: 1.6;
        }
        
        .stNumberInput > div > div > input,
        .stTextInput > div > div > input {
            font-size: 17px;
            padding: 14px;
            height: auto;
        }
        
        .dataframe {
            font-size: 16px;
        }
        
        .dataframe th, .dataframe td {
            font-size: 16px;
            padding: 14px 10px;
        }
        
        .stMarkdown {
            font-size: 17px;
            line-height: 1.6;
            margin-bottom: 0.25rem;
        }
        
        h1, h2, h3, h4 {
            font-size: 1.5em;
            line-height: 1.3;
        }
        
        /* Make sidebar full width on mobile */
        .css-1d391kg {
            width: 100%;
            height: auto;
            position: relative;
        }
        
        /* Adjust main content padding */
        .block-container {
            padding: 1rem;
        }
        
        /* Make charts responsive */
        .element-container {
            width: 100%;
            overflow-x: auto;
            margin-bottom: 0.5rem;
        }
    }
    
    /* Alerts and info boxes */
    .stAlert {
        background-color: #2d2d2d;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.25rem 0;
        border: 1px solid #3d3d3d;
        font-size: 16px;
        line-height: 1.6;
    }
    
    .stMarkdown {
        background-color: #2d2d2d;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.25rem 0;
        line-height: 1.6;
        font-size: 16px;
    }
    
    /* Streamlit specific elements - Strong overrides */
    div[data-testid="stMarkdown"] h2 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    div[data-testid="stDataFrame"] {
        background-color: #2d2d2d !important;
    }

    div[data-testid="stDataFrame"] th {
        background-color: #3d3d3d !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    div[data-testid="stDataFrame"] td {
        color: #ffffff !important;
    }

    /* Override Streamlit's default text colors */
    div[data-testid="stMarkdown"] {
        color: #ffffff !important;
    }

    div[data-testid="stMarkdown"] p {
        color: #ffffff !important;
    }

    div[data-testid="stMarkdown"] strong {
        color: #ffffff !important;
    }

    /* Override Streamlit's table styling */
    .stDataFrame {
        background-color: #2d2d2d !important;
    }

    .stDataFrame th {
        background-color: #3d3d3d !important;
        color: #ffffff !important;
    }

    .stDataFrame td {
        color: #ffffff !important;
    }

    /* Force all text to be white except inputs */
    * {
        color: #ffffff !important;
    }
    
    /* Improve contrast for better readability */
    .stMarkdown, .stAlert, .dataframe {
        color: #ffffff !important;  /* Force white color with !important */
    }
    
    /* Improve button text readability */
    .stButton > button, .stDownloadButton > button {
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    
    /* Charts */
    .element-container {
        background-color: #2d2d2d;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.25rem 0;
        width: 100%;
        overflow-x: auto;
    }
    
    /* Hover effects */
    .dataframe tr:hover {
        background-color: #3d3d3d;
    }
    
    /* Total row styling */
    .dataframe tr:last-child {
        font-weight: bold;
        background-color: #3d3d3d;
    }
    
    /* Select boxes */
    .stSelectbox > div > div > div {
        background-color: #2d2d2d;
        color: white;
        padding: 8px;
        border-radius: 8px;
        border: 1px solid #3d3d3d;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background-color: #2d2d2d;
        padding: 8px;
    }
    
    /* Checkboxes */
    .stCheckbox > div > div > div {
        background-color: #2d2d2d;
        padding: 8px;
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background-color: #003366;
        color: white;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 0.25rem 0;
        width: 100%;
        font-size: 16px;
        border: none;
        -webkit-appearance: none;
    }
    
    /* Improve touch targets */
    button, input, select, textarea {
        touch-action: manipulation;
    }
    
    /* Prevent text selection */
    .stButton > button, .stDownloadButton > button {
        user-select: none;
        -webkit-user-select: none;
    }
    
    /* Improve scrolling */
    .element-container, .dataframe {
        -webkit-overflow-scrolling: touch;
    }
    
    /* Reduce spacing between form elements */
    .stForm {
        margin-bottom: 0.25rem;
    }
    
    /* Reduce spacing in number input widgets */
    .stNumberInput > div > div > div {
        padding: 0.25rem;
    }
    
    /* Reduce spacing in text input widgets */
    .stTextInput > div > div > div {
        padding: 0.25rem;
    }
    
    /* Reduce spacing in select boxes */
    .stSelectbox > div > div > div {
        padding: 0.25rem;
    }
    
    /* Reduce spacing in sliders */
    .stSlider > div > div > div {
        padding: 0.25rem;
    }
    
    /* Reduce spacing in checkboxes */
    .stCheckbox > div > div > div {
        padding: 0.25rem;
    }

    /* Streamlit input widgets - Force black text */
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        background-color: #ffffff !important;
        opacity: 1 !important;
    }

    /* Input labels */
    div[data-testid="stNumberInput"] label,
    div[data-testid="stTextInput"] label {
        color: #ffffff !important;
    }

    /* Input containers */
    div[data-testid="stNumberInput"] > div,
    div[data-testid="stTextInput"] > div {
        background-color: #ffffff !important;
    }

    /* Placeholder text */
    div[data-testid="stNumberInput"] input::placeholder,
    div[data-testid="stTextInput"] input::placeholder {
        color: #666666 !important;
        -webkit-text-fill-color: #666666 !important;
        opacity: 1 !important;
    }

    /* Override any Streamlit default styles */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        background-color: #ffffff !important;
        opacity: 1 !important;
    }

    /* iOS specific overrides */
    @supports (-webkit-touch-callout: none) {
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input {
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            background-color: #ffffff !important;
            opacity: 1 !important;
        }
    }
    </style>
""",
    unsafe_allow_html=True,
)

@dataclass
class LeaseInputs:
    """
    A dataclass representing all inputs required for lease modeling.
    
    Attributes:
        year1_rent_psf (float): First year rent per square foot
        lease_term_months (int): Total lease term in months
        annual_escalation_pct (float): Annual rent escalation percentage
        rentable_sf (float): Total rentable square footage
        free_rent_months (int): Number of months of free rent
        ti_psf (float): Tenant improvement allowance per square foot
        lc_psf (float): Leasing commission per square foot
        discount_rate (float): Discount rate for NPV calculations
    """
    year1_rent_psf: float
    lease_term_months: int
    annual_escalation_pct: float
    rentable_sf: float
    free_rent_months: int
    ti_psf: float
    lc_psf: float
    discount_rate: float

    def validate(self) -> Tuple[bool, str]:
        """
        Validates the lease inputs for logical consistency.
        
        Returns:
            Tuple[bool, str]: A tuple containing (is_valid, error_message)
        """
        try:
            if self.lease_term_months <= 0:
                return False, "Lease term must be positive"
            if self.lease_term_months > MAX_LEASE_TERM_YEARS * MONTHS_PER_YEAR:
                return False, f"Lease term cannot exceed {MAX_LEASE_TERM_YEARS} years"
            if self.free_rent_months >= self.lease_term_months:
                return False, "Free rent period cannot exceed lease term"
            if self.free_rent_months > MAX_FREE_RENT_MONTHS:
                return False, f"Free rent period cannot exceed {MAX_FREE_RENT_MONTHS} months"
            if self.rentable_sf <= 0:
                return False, "Rentable square footage must be positive"
            if self.year1_rent_psf < 0:
                return False, "Rent PSF cannot be negative"
            if self.annual_escalation_pct < 0:
                return False, "Escalation rate cannot be negative"
            if self.discount_rate < 0:
                return False, "Discount rate cannot be negative"
            if self.ti_psf < 0:
                return False, "TI allowance cannot be negative"
            if self.lc_psf < 0:
                return False, "Leasing commission cannot be negative"
            return True, ""
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False, f"Validation error: {str(e)}"

def format_currency(value: Union[float, int, str]) -> str:
    """
    Formats a numeric value as currency.
    
    Args:
        value: The value to format
        
    Returns:
        str: Formatted currency string
    """
    try:
        if isinstance(value, str):
            return value
        return f"${float(value):,.2f}"
    except (ValueError, TypeError):
        return str(value)

def get_escalated_rent_per_sf(
    base_rent: float, escalation_pct: float, month: int
) -> float:
    """
    Calculates the escalated rent per square foot for a given month.
    
    Args:
        base_rent (float): Base rent per square foot
        escalation_pct (float): Annual escalation percentage
        month (int): Month number (0-based)
        
    Returns:
        float: Escalated rent per square foot for the given month
    """
    year = month // MONTHS_PER_YEAR
    return (base_rent * (1 + escalation_pct / 100) ** year) / MONTHS_PER_YEAR


def calculate_monthly_rents(inputs: LeaseInputs) -> List[float]:
    """
    Calculates the monthly rent payments for the entire lease term.
    
    Args:
        inputs (LeaseInputs): Lease input parameters
        
    Returns:
        List[float]: List of monthly rent payments
    """
    try:
        return [
            0
            if month < inputs.free_rent_months
            else get_escalated_rent_per_sf(
                inputs.year1_rent_psf, inputs.annual_escalation_pct, month
            )
            * inputs.rentable_sf
            for month in range(inputs.lease_term_months)
        ]
    except Exception as e:
        logger.error(f"Error calculating monthly rents: {str(e)}")
        raise


def calculate_npv(cash_flows: List[float], annual_discount_rate: float) -> float:
    """
    Calculates the Net Present Value (NPV) of a series of cash flows.
    
    Args:
        cash_flows (List[float]): List of monthly cash flows
        annual_discount_rate (float): Annual discount rate in percentage
        
    Returns:
        float: Net Present Value
    """
    try:
        monthly_rate = (1 + annual_discount_rate / 100) ** (1 / MONTHS_PER_YEAR) - 1
        return sum(
            cf / ((1 + monthly_rate) ** i) for i, cf in enumerate(cash_flows, start=1)
        )
    except Exception as e:
        logger.error(f"Error calculating NPV: {str(e)}")
        raise


def calculate_payback_period(
    cash_flows: List[float], deal_cost: float
) -> Optional[int]:
    """
    Calculates the payback period in months for a given investment.
    
    Args:
        cash_flows (List[float]): List of monthly cash flows
        deal_cost (float): Total initial investment cost
        
    Returns:
        Optional[int]: Payback period in months, or None if payback is not achieved
    """
    try:
        cumulative_cash = 0
        for i, cf in enumerate(cash_flows, start=1):
            cumulative_cash += cf
            if cumulative_cash >= deal_cost:
                return i
        return None
    except Exception as e:
        logger.error(f"Error calculating payback period: {str(e)}")
        raise


def summarize_annual_rent(
    monthly_rents: List[float], inputs: LeaseInputs
) -> Dict[int, Dict[str, float]]:
    """
    Creates an annual summary of rent payments, including gross rent, abatement, and net rent.
    
    Args:
        monthly_rents (List[float]): List of monthly rent payments
        inputs (LeaseInputs): Lease input parameters
        
    Returns:
        Dict[int, Dict[str, float]]: Annual summary with metrics per year
    """
    try:
        annual_summary = defaultdict(
            lambda: {"gross": 0, "abatement": 0, "net": 0, "gross_psf": 0, "net_psf": 0}
        )
        escalator = 1 + inputs.annual_escalation_pct / 100
        
        for month, rent_paid in enumerate(monthly_rents):
            year = (month // MONTHS_PER_YEAR) + 1
            base_monthly_rent = (
                (inputs.year1_rent_psf * (escalator ** (year - 1))) / MONTHS_PER_YEAR
            ) * inputs.rentable_sf
            
            annual_summary[year]["gross"] += base_monthly_rent
            if rent_paid == 0:
                annual_summary[year]["abatement"] += base_monthly_rent
            else:
                annual_summary[year]["net"] += rent_paid
                
        for year, data in annual_summary.items():
            data["gross_psf"] = data["gross"] / inputs.rentable_sf
            data["net_psf"] = data["net"] / inputs.rentable_sf
            
        return annual_summary
    except Exception as e:
        logger.error(f"Error summarizing annual rent: {str(e)}")
        raise


def plot_monthly_rents(
    inputs: LeaseInputs, monthly_rents: List[float], deal_cost: float
) -> None:
    """
    Creates an interactive plot showing monthly rent PSF and cumulative rent.
    
    Args:
        inputs (LeaseInputs): Lease input parameters
        monthly_rents (List[float]): List of monthly rent payments
        deal_cost (float): Total initial investment cost
    """
    try:
        months = list(range(1, inputs.lease_term_months + 1))
        cumulative_rent = np.cumsum(monthly_rents)
        escalator = 1 + inputs.annual_escalation_pct / 100
        psf_rents = [
            inputs.year1_rent_psf * (escalator ** (m // 12))
            for m in range(inputs.lease_term_months)
        ]

        fig = go.Figure()

        # Rent PSF line
        fig.add_trace(
            go.Scatter(
                x=months,
                y=psf_rents,
                mode="lines",
                name="Annual Rent PSF",
                line=dict(color="#003399", width=4, shape="hv"),
                hovertemplate="Month %{x}<br>Rent PSF: $%{y:.2f}",
            )
        )

        # Cumulative rent line
        fig.add_trace(
            go.Scatter(
                x=months,
                y=cumulative_rent,
                mode="lines",
                name="Cumulative Rent",
                line=dict(color="#2e8b57", width=3, dash="dash"),
                yaxis="y2",
                hovertemplate="Month %{x}<br>Cumulative Rent: $%{y:,.0f}",
            )
        )

        # Deal cost line
        fig.add_hline(
            y=deal_cost,
            line=dict(color="red", width=3, dash="dot"),
            annotation_text="TI + LC Cost",
            annotation_position="top right",
            annotation_font_size=10,
            yref="y2",
        )

        # Breakeven marker
        breakeven_month = next(
            (i + 1 for i, val in enumerate(cumulative_rent) if val >= deal_cost), None
        )
        if breakeven_month:
            fig.add_vline(
                x=breakeven_month,
                line=dict(color="yellow", width=2, dash="dash"),
                annotation_text="Breakeven",
                annotation_position="bottom right",
            )

        # Escalation step markers
        for year in range(0, inputs.lease_term_months // 12):
            month_marker = year * 12 + 1

            fig.add_shape(
                type="line",
                x0=month_marker,
                x1=month_marker,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="gray", width=1, dash="dot"),
            )

            fig.add_annotation(
                x=month_marker + 6,
                y=1.05,
                yref="paper",
                text=f"Year {year + 1}<br>${inputs.year1_rent_psf * (escalator**year):,.2f} PSF",
                showarrow=False,
                font=dict(size=10),
                xanchor="center",
            )

        # Highlight free rent period
        if inputs.free_rent_months > 0:
            fig.add_vrect(
                x0=1,
                x1=inputs.free_rent_months + 1,
                fillcolor="lightgray",
                opacity=0.3,
                layer="below",
                line_width=0,
                annotation_text="Free Rent Period",
                annotation_position="top left",
            )

        # Layout formatting
        fig.update_layout(
            template="plotly_dark",
            title="Rent PSF and Cumulative Rent",
            xaxis=dict(title="Month"),
            yaxis=dict(
                rangemode="tozero",
                title="Rent PSF ($)",
                side="left",
                color="#1a1a2e",
                showgrid=True,
            ),
            yaxis2=dict(
                title="Cumulative Rent ($)",
                overlaying="y",
                side="right",
                color="#2e8b57",
                showgrid=False,
            ),
            legend=dict(
                x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)", bordercolor="gray"
            ),
            margin=dict(l=40, r=40, t=50, b=40),
            height=650,
            width=1000,
        )

        fig.update_layout(width=1200, height=650)
        st.plotly_chart(fig, use_container_width=False)
    except Exception as e:
        logger.error(f"Error creating rent plot: {str(e)}")
        st.error("Error creating rent visualization. Please check your inputs.")


def display_lease_summary(inputs: LeaseInputs, monthly_rents: List[float], total_cost: float) -> None:
    """
    Displays the lease summary table with key metrics.
    
    Args:
        inputs (LeaseInputs): Lease input parameters
        monthly_rents (List[float]): List of monthly rent payments
        total_cost (float): Total initial investment cost
    """
    try:
        gross_rent = sum(monthly_rents)
        net_rent = gross_rent
        ti = inputs.ti_psf * inputs.rentable_sf
        lc = inputs.lc_psf * inputs.rentable_sf
        npv = calculate_npv(monthly_rents, inputs.discount_rate)
        payback_month = calculate_payback_period(monthly_rents, total_cost)
        lease_term_years = inputs.lease_term_months / 12
        net_effective_rent_psf = (gross_rent / inputs.rentable_sf) / lease_term_years

        st.subheader("Lease Summary")
        sum_table = pd.DataFrame(
            {
                "Metric": [
                    "Rentable SF",
                    "Lease Term (Months)",
                    "Year 1 Rent PSF",
                    "Free Rent (Months)",
                    "Annual Escalation (%)",
                    "Total Gross Rent",
                    "Tenant Improvements (TI)",
                    "Leasing Commissions (LC)",
                    "Total Deal Cost",
                    "Net Rent",
                    "Net Effective Rent PSF (Annualized)",
                    "NPV",
                    "Payback Period",
                ],
                "Amount ($)": [
                    f"{inputs.rentable_sf:,.0f} SF",
                    f"{inputs.lease_term_months} months",
                    f"${inputs.year1_rent_psf:,.2f}",
                    f"{inputs.free_rent_months} months",
                    f"{inputs.annual_escalation_pct:.2f}%",
                    f"${gross_rent:,.2f}",
                    f"${ti:,.2f}",
                    f"${lc:,.2f}",
                    f"${total_cost:,.2f}",
                    f"${net_rent:,.2f}",
                    f"${net_effective_rent_psf:,.2f}",
                    f"${npv:,.2f}",
                    f"{'Not achieved' if not payback_month else f'{payback_month:,} months (~{payback_month / 12:.1f} years)'}",
                ],
                "$/SF": [
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    f"${gross_rent / inputs.rentable_sf:,.2f}",
                    f"${inputs.ti_psf:,.2f}",
                    f"${inputs.lc_psf:,.2f}",
                    f"${total_cost / inputs.rentable_sf:,.2f}",
                    f"${net_rent / inputs.rentable_sf:,.2f}",
                    f"${net_effective_rent_psf:,.2f}",
                    "-",
                    "-",
                ],
            }
        )
        st.table(sum_table)
    except Exception as e:
        logger.error(f"Error displaying lease summary: {str(e)}")
        st.error("Error displaying lease summary. Please check your inputs.")

def display_annual_summary(monthly_rents: List[float], inputs: LeaseInputs) -> None:
    """
    Displays the annual summary table and provides download options.
    
    Args:
        monthly_rents (List[float]): List of monthly rent payments
        inputs (LeaseInputs): Lease input parameters
    """
    try:
        # Create annual summary
        annual_summary = summarize_annual_rent(monthly_rents, inputs)
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(annual_summary, orient="index").round(2)
        df.index = df.index.astype(str)  # Convert index to strings
        
        # Rename columns
        df = df.rename(
            columns={
                "gross": "Gross Rent",
                "abatement": "Free Rent Abatement",
                "net": "Net Rent",
                "gross_psf": "Gross Rent PSF",
                "net_psf": "Net Rent PSF",
            }
        )
        
        # Add monthly rent column
        df["Net Rent / Month"] = (df["Net Rent"] / 12).round(2)
        
        # Calculate totals
        total_data = {
            "Gross Rent": df["Gross Rent"].sum(),
            "Free Rent Abatement": df["Free Rent Abatement"].sum(),
            "Net Rent": df["Net Rent"].sum(),
            "Gross Rent PSF": df["Gross Rent"].sum() / inputs.rentable_sf,
            "Net Rent PSF": df["Net Rent"].sum() / inputs.rentable_sf,
            "Net Rent / Month": df["Net Rent"].sum() / (12 * len(df))
        }
        
        # Create total row with string index
        total_df = pd.DataFrame([total_data], index=["Total"])
        
        # Combine DataFrames
        df = pd.concat([df, total_df])
        
        # Format numeric columns
        numeric_columns = ["Gross Rent", "Free Rent Abatement", "Net Rent", "Gross Rent PSF", "Net Rent PSF", "Net Rent / Month"]
        for col in numeric_columns:
            df[col] = df[col].apply(format_currency)

        # Display styled dataframe
        st.subheader("Annual Summary")
        
        # Display the table using st.dataframe instead of HTML
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=False,
            column_config={
                "Gross Rent": st.column_config.NumberColumn(
                    "Gross Rent",
                    format="$%.2f"
                ),
                "Free Rent Abatement": st.column_config.NumberColumn(
                    "Free Rent Abatement",
                    format="$%.2f"
                ),
                "Net Rent": st.column_config.NumberColumn(
                    "Net Rent",
                    format="$%.2f"
                ),
                "Gross Rent PSF": st.column_config.NumberColumn(
                    "Gross Rent PSF",
                    format="$%.2f"
                ),
                "Net Rent PSF": st.column_config.NumberColumn(
                    "Net Rent PSF",
                    format="$%.2f"
                ),
                "Net Rent / Month": st.column_config.NumberColumn(
                    "Net Rent / Month",
                    format="$%.2f"
                )
            }
        )

        # Create export data
        export_df = df.copy()
        
        # Remove currency formatting for export
        for col in numeric_columns:
            export_df[col] = export_df[col].str.replace('$', '').str.replace(',', '').astype(float)

        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download as CSV",
                export_df.to_csv(),
                "lease_summary.csv",
                "text/csv",
                help="Download annual summary data in CSV format"
            )
        with col2:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                export_df.to_excel(writer, index=True, sheet_name="Annual Summary")
            st.download_button(
                "Download as Excel",
                excel_buffer.getvalue(),
                "lease_summary.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download annual summary data in Excel format"
            )
    except Exception as e:
        logger.error(f"Error displaying annual summary: {str(e)}")
        st.error(f"Error displaying annual summary: {str(e)}")

def main():
    """
    Main function that runs the Streamlit application.
    """
    st.title("Lease Modeling Dashboard")
    st.markdown("""
        This dashboard helps you model commercial real estate leases by calculating various financial metrics,
        including Net Present Value (NPV), payback period, and annual rent summaries.
    """)
    
    # Initialize session state for inputs if not exists
    if 'inputs' not in st.session_state:
        st.session_state.inputs = None
    
    # Create input form in sidebar
    with st.sidebar:
        st.header("Input Parameters")
        with st.form(key="input_form"):
            year1_rent_psf = st.number_input(
                "Year 1 Rent PSF",
                min_value=0.0,
                value=50.0,
                step=1.0,
                help="First-year rent per square foot",
                format="%.2f"
            )
            lease_term_months = st.number_input(
                "Lease Term (Months)",
                min_value=1,
                value=120,
                step=12,
                help=f"Lease duration in months (max {MAX_LEASE_TERM_YEARS} years)",
                format="%d"
            )
            annual_escalation_pct = st.number_input(
                "Annual Escalation (%)",
                min_value=0.0,
                value=DEFAULT_ESCALATION_RATE,
                step=0.5,
                help="Annual rent increase percentage",
                format="%.2f"
            )
            rentable_sf = st.number_input(
                "Rentable SF",
                min_value=1.0,
                value=10000.0,
                step=1000.0,
                format="%.0f",
                help="Total rentable square footage"
            )
            free_rent_months = st.number_input(
                "Free Rent (Months)",
                min_value=0,
                value=3,
                step=1,
                help=f"Initial rent abatement period (max {MAX_FREE_RENT_MONTHS} months)",
                format="%d"
            )
            ti_psf = st.number_input(
                "TI Allowance ($/SF)",
                min_value=0.0,
                value=50.0,
                step=5.0,
                help="Tenant improvement allowance per SF",
                format="%.2f"
            )
            lc_psf = st.number_input(
                "Leasing Commission ($/SF)",
                min_value=0.0,
                value=20.0,
                step=1.0,
                help="Broker commissions per SF",
                format="%.2f"
            )
            discount_rate = st.number_input(
                "Discount Rate (%)",
                min_value=0.0,
                value=DEFAULT_DISCOUNT_RATE,
                step=0.5,
                help="Discount rate for NPV calculations",
                format="%.2f"
            )
            
            submitted = st.form_submit_button("Update Model")
            
            if submitted:
                inputs = LeaseInputs(
                    year1_rent_psf,
                    lease_term_months,
                    annual_escalation_pct,
                    rentable_sf,
                    free_rent_months,
                    ti_psf,
                    lc_psf,
                    discount_rate,
                )
                
                is_valid, error_msg = inputs.validate()
                if not is_valid:
                    st.error(f"Invalid inputs: {error_msg}")
                else:
                    st.session_state.inputs = inputs
                    st.success("Model updated successfully!")

    # Main content area
    if st.session_state.inputs is not None:
        try:
            inputs = st.session_state.inputs
            # Calculate monthly rents and total cost
            monthly_rents = calculate_monthly_rents(inputs)
            total_cost = (inputs.ti_psf + inputs.lc_psf) * inputs.rentable_sf
            
            # Display lease summary
            display_lease_summary(inputs, monthly_rents, total_cost)
            
            # Display rent chart
            st.subheader("Monthly & Cumulative Rent Chart")
            plot_monthly_rents(inputs, monthly_rents, total_cost)
            
            # Display annual summary
            display_annual_summary(monthly_rents, inputs)
            
            # Footer
            st.markdown("<hr style='margin-top: 2rem;'>", unsafe_allow_html=True)
            st.markdown(
                "<p style='text-align:center; font-size: 13px; color: gray;'>Created by Michael Morretta</p>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            logger.error(f"Error in main application: {str(e)}")
            st.error("An error occurred while processing your request. Please check your inputs and try again.")
    else:
        st.info("Please enter the lease parameters in the sidebar and click 'Update Model' to see the results.")

if __name__ == "__main__":
    main()
