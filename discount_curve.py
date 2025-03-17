#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
from dataclasses import dataclass
import utils
import abstract_curve

# DiscountCurve class
# OISDeposit class
# OISSwap class


# In[29]:


from abstract_curve import InterestRateCurve
from utils import MathUtils, Interpolator, LogLinearInterpolator, NelsonSiegelModel


# In[31]:


class DiscountCurve(InterestRateCurve):
    """
    Discount factor curve
    Primary use: OIS (SOFR/ESTR/SONIA) discount curve for risk-free discounting
    """
    
    def __init__(self, interpolator: Interpolator = None, model: NelsonSiegelModel = None):
        self.discount_factors = {}  # {tenor: discount_factor}
        self.interpolator = interpolator or LogLinearInterpolator()
        self.model = model
        self.use_model = False
        
    def add_discount_factor(self, tenor: float, df: float):
        """Add a market-implied discount factor"""
        self.discount_factors[tenor] = df
        
    def get_discount_factor(self, t: float):
        """
        Get discount factor for tenor t
        Uses direct lookup, model, or interpolation with fallback to extrapolation
        """
        if t <= 0:
            return 1.0

        if t in self.discount_factors:
            return self.discount_factors[t]

        if self.use_model and self.model is not None:
            # Convert model-implied zero rate to discount factor
            r = self.model.get_value(t)
            return MathUtils.calculate_df_from_zero_rate(r, t)

        # Use interpolation between known points
        tenors = sorted(self.discount_factors.keys())
        dfs = [self.discount_factors[tenor] for tenor in tenors]

        try:
            return self.interpolator.interpolate(t, tenors, dfs)
        except ValueError as e:
            # Fallback extrapolation if interpolation fails
            if t < min(tenors):
                # Short end - use first value
                return self.discount_factors[tenors[0]]
            elif t > max(tenors):
                # Long end - use simple extrapolation
                max_tenor = max(tenors)
                max_df = self.discount_factors[max_tenor]
                # Get implied zero rate at max tenor
                r = -np.log(max_df) / max_tenor
                # Apply slight upward slope (5 bps per year)
                r_t = r + 0.0005 * (t - max_tenor)
                # Calculate new DF
                return np.exp(-r_t * t)
            else:
                # Shouldn't happen, but just in case
                raise ValueError(f"Cannot interpolate discount factor at {t}: {e}")
        
    def get_zero_rate(self, t: float):
        """Convert discount factor to continuously compounded zero rate"""
        if t <= 0:
            return 0.0
        df = self.get_discount_factor(t)
        return MathUtils.calculate_zero_rate_from_df(df, t)
        
    def apply_model(self):
        """
        Apply Nelson-Siegel model to smooth the curve
        Converts discount factors to zero rates, calibrates model
        """
        if self.model is None:
            self.model = NelsonSiegelModel()
            
        # Convert DFs to zero rates for calibration
        tenors = sorted(self.discount_factors.keys())
        zero_rates = [self.get_zero_rate(t) for t in tenors]
        
        # Calibrate model to zero rates
        self.model.calibrate(tenors, zero_rates)
        self.use_model = True
        
    def _get_max_tenor(self):
        """Get maximum tenor in the curve"""
        if self.discount_factors:
            return max(self.discount_factors.keys())
        return 10.0
        
    def __repr__(self):
        lines = ["DiscountCurve:"]
        for tenor in sorted(self.discount_factors.keys()):
            df = self.discount_factors[tenor]
            zr = self.get_zero_rate(tenor) * 100  # To percent
            lines.append(f"  T={tenor:.2f} -> DF={df:.8f}, ZR={zr:.4f}%")
        if self.use_model:
            lines.append("  (Nelson-Siegel model applied)")
        return "\n".join(lines)


# In[32]:


@dataclass
class OISDeposit:
    """Overnight Index Swap Deposit"""
    maturity: float
    rate: float
    
    def bootstrap(self, curve: DiscountCurve):
        """
        Bootstrap OIS deposit into discount curve
        Reuses MathUtils to maintain consistency with pricing
        """
        # For deposits, the forward rate equals the deposit rate
        # DF(0,T) = 1/(1 + r*T)
        df = 1.0 / (1.0 + self.rate * self.maturity)
        curve.add_discount_factor(self.maturity, df)
    
    def __repr__(self):
        return f"OISDeposit(T={self.maturity:.2f}, r={self.rate*100:.4f}%)"


# In[33]:


@dataclass
class OISSwap:
    """Overnight Index Swap"""
    maturity: float
    rate: float
    payment_frequency: int = 4  # Quarterly by default
    
    def bootstrap(self, curve: DiscountCurve):
        """
        Bootstrap OIS swap into discount curve
        Reuses MathUtils to maintain consistency with pricing
        """
        # Generate payment schedule
        payment_times = []
        payment_interval = 1.0 / self.payment_frequency
        current_time = payment_interval
        while current_time <= self.maturity + 1e-10:
            payment_times.append(current_time)
            current_time += payment_interval
            
        # Create cash flows for fixed leg
        fixed_cash_flows = []
        prev_time = 0.0
        for t in payment_times:
            period = t - prev_time
            coupon = self.rate * period
            fixed_cash_flows.append(coupon)
            prev_time = t
            
        # Get known discount factors
        known_dfs = {t: curve.get_discount_factor(t) for t in payment_times[:-1]}
        known_dfs[0.0] = 1.0
        
        # For a par swap, PV(fixed leg) = 1 - DF(T)
        # With simplified floating leg approximation
        target_pv = 1.0  # Normalized to 1.0 notional
        
        # Solve for the final discount factor
        final_df = MathUtils.solve_for_df(
            payment_times,
            fixed_cash_flows,
            known_dfs,
            target_pv
        )
        
        # Add to curve
        curve.add_discount_factor(payment_times[-1], final_df)
    
    def __repr__(self):
        freq_name = {1: "annual", 2: "semi-annual", 4: "quarterly", 12: "monthly"}.get(
            self.payment_frequency, f"{self.payment_frequency}/year")
        return f"OISSwap(T={self.maturity:.2f}, r={self.rate*100:.4f}%, {freq_name})"


# In[34]:


get_ipython().system('jupyter nbconvert --to script discount_curve.ipynb')

