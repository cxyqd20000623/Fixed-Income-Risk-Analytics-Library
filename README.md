# Fixed-Income-Risk-Analytics-Library
Fixed Income Library
A comprehensive Python library for fixed income analysis, yield curve construction, and financial instrument pricing.
Overview
This Fixed Income Library provides a robust framework for fixed income analytics, including:

Yield curve construction (discount curves, zero curves, forward curves)
Multi-curve framework with OIS discounting
Financial instrument pricing (bonds, swaps, deposits, FRAs)
Risk analytics (duration, convexity, DV01)
Curve smoothing using Nelson-Siegel models

The library is designed with modern fixed income practices in mind, supporting the post-2008 multi-curve framework where different curves are used for discounting (typically OIS) and forward rate projection.
Key Features

Flexible Yield Curve Construction

OIS discount curves for risk-free discounting
Forward curves for projection (SOFR, EURIBOR, etc.)
Treasury/government zero curves
Bootstrapping from market instruments


Financial Instrument Pricing

Fixed-rate bonds (government and corporate)
Interest rate swaps (fixed-for-floating)
OIS deposits and swaps
Forward Rate Agreements (FRAs)


Risk Measures

Macaulay and modified duration
Effective duration for swaps
Convexity
DV01/PV01 sensitivity metrics


Advanced Functionality

Nelson-Siegel curve smoothing
Interpolation methods (log-linear, monotonic cubic)
Forward rate calculations
Yield conversion utilities



Structure
The library is organized into several modules:

abstract_curve.py - Base classes for interest rate curves
discount_curve.py - OIS discount curve and related instruments
zero_curve.py - Zero rate curves and treasury instruments
forward_curve.py - Forward curves and IBOR/SOFR instruments
fixed_income.py - Fixed income pricing and risk calculation
curve_builders.py - Utilities for constructing curves from market data
utils.py - Mathematical utilities and interpolation methods
