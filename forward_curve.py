#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
from dataclasses import dataclass
from abstract_curve import InterestRateCurve
from utils import MathUtils, Interpolator, MonotonicCubicInterpolator, NelsonSiegelModel
from discount_curve import DiscountCurve

# ForwardCurve class
# IBORDeposit class
# FRA class
# IBORSwap class


# In[12]:


class ForwardCurve(InterestRateCurve):
    """
    Forward rate curve for a specific index (e.g., 3M SOFR)
    Primarily used for projecting floating rate payments
    """
    
    def __init__(self, index_name: str = "SOFR", tenor_basis: float = 0.25, 
                 discount_curve: InterestRateCurve = None,
                 interpolator: Interpolator = None, model: NelsonSiegelModel = None):
        """
        Initialize forward curve
        
        Args:
            index_name: Name of the index (e.g., "SOFR", "EURIBOR")
            tenor_basis: Tenor of the index in years (0.25 = 3M)
            discount_curve: Curve for discounting (typically OIS)
            interpolator: Interpolator for forward rates
            model: Curve smoothing model
        """
        self.index_name = index_name
        self.tenor_basis = tenor_basis
        self.discount_curve = discount_curve
        self.forward_rates = {}  # {start_tenor: forward_rate}
        self.interpolator = interpolator or MonotonicCubicInterpolator()
        self.interpolator_needs_update = True
        self.model = model
        self.use_model = False
        
    def add_forward_rate(self, start_tenor: float, rate: float) -> None:
        """Add forward rate to curve"""
        self.forward_rates[start_tenor] = rate
        self.interpolator_needs_update = True
        
    def _update_interpolator(self) -> None:
        """Update the interpolator with current data"""
        if not self.interpolator_needs_update:
            return
            
        if not isinstance(self.interpolator, MonotonicCubicInterpolator):
            return
            
        starts = sorted(self.forward_rates.keys())
        rates = [self.forward_rates[s] for s in starts]
        
        if len(starts) >= 2:
            self.interpolator.fit(starts, rates)
            self.interpolator_needs_update = False
        
    def get_forward_rate(self, start: float, end: float = None) -> float:
        """Get forward rate for the specified period"""
        # Default end tenor
        if end is None:
            end = start + self.tenor_basis
            
        # Check if it matches the standard tenor
        if abs(end - start - self.tenor_basis) < 1e-6:
            if start in self.forward_rates:
                return self.forward_rates[start]
                
            if self.use_model and self.model is not None:
                return self.model.get_value(start)
                
            # Interpolate
            try:
                if isinstance(self.interpolator, MonotonicCubicInterpolator):
                    self._update_interpolator()
                    return self.interpolator.interpolate(start)
                else:
                    starts = sorted(self.forward_rates.keys())
                    rates = [self.forward_rates[s] for s in starts]
                    return self.interpolator.interpolate(start, starts, rates)
            except ValueError:
                # Handle extrapolation
                starts = sorted(self.forward_rates.keys())
                if start < starts[0]:
                    return self.forward_rates[starts[0]]
                elif start > starts[-1]:
                    return self.forward_rates[starts[-1]]
        else:
                    raise ValueError(f"Cannot interpolate forward rate at {start}")
                
        # Non-standard tenor - use discount curve if available
        if self.discount_curve:
            return self.discount_curve.get_forward_rate(start, end)
            
        raise ValueError(f"Cannot calculate forward rate for [{start}, {end}]")
        
    def get_discount_factor(self, t: float) -> float:
        """Get discount factor using reference discount curve"""
        if self.discount_curve:
            return self.discount_curve.get_discount_factor(t)
            
        # Without a discount curve, convert from forward rates (simplified)
        if t <= 0:
            return 1.0
            
        # Build discount factor from compounding forward rates
        df = 1.0
        current_t = 0.0
        while current_t < t:
            next_t = min(current_t + self.tenor_basis, t)
            dt = next_t - current_t
            try:
                fwd_rate = self.get_forward_rate(current_t, next_t)
                df *= 1.0 / (1.0 + fwd_rate * dt)
            except ValueError:
                raise ValueError(f"Cannot construct DF({t}) from forwards")
            current_t = next_t
            
        return df
    
    def get_zero_rate(self, t: float) -> float:
        """Calculate zero rate from discount factor"""
        if t <= 0:
            return 0.0
        df = self.get_discount_factor(t)
        return MathUtils.calculate_zero_rate_from_df(df, t)
        
    def apply_model(self) -> None:
        """Apply Nelson-Siegel smoothing"""
        if self.model is None:
            self.model = NelsonSiegelModel()
            
        starts = sorted(self.forward_rates.keys())
        rates = [self.forward_rates[s] for s in starts]
        
        self.model.calibrate(starts, rates)
        self.use_model = True
        
    def _get_max_tenor(self) -> float:
        """Get maximum tenor in the curve"""
        if self.forward_rates:
            return max(self.forward_rates.keys()) + self.tenor_basis
        return 10.0
        
    def __repr__(self) -> str:
        lines = [f"{int(self.tenor_basis*12)}M {self.index_name} ForwardCurve:"]
        for start in sorted(self.forward_rates.keys()):
            end = start + self.tenor_basis
            rate = self.forward_rates[start] * 100  # To percent
            lines.append(f"  [{start:.2f}, {end:.2f}] -> {rate:.4f}%")
        if self.use_model:
            lines.append("  (Nelson-Siegel model applied)")
        return "\n".join(lines)


# In[13]:


@dataclass
class IBORDeposit:
    """IBOR or SOFR Deposit"""
    maturity: float
    rate: float
    index_name: str = "SOFR"
    
    def bootstrap(self, curve: ForwardCurve) -> None:
        """
        Bootstrap IBOR deposit into forward curve
        Uses discount factors from SOFR Deposit to calculate short-term forward rates
        """
        # Calculate discount factor from deposit rate
        df = 1.0 / (1.0 + self.rate * self.maturity)
        
        # For short-term deposits covering exactly one period
        if abs(self.maturity - curve.tenor_basis) < 1e-6:
            # Direct conversion - forward rate equals deposit rate
            curve.add_forward_rate(0.0, self.rate)
        elif self.maturity > curve.tenor_basis:
            # For longer deposits, need to calculate implied forward rate
            # First, check if we need intermediate points
            num_periods = int(self.maturity / curve.tenor_basis)
            if num_periods * curve.tenor_basis < self.maturity - 1e-10:
                raise ValueError(f"Deposit maturity {self.maturity} not divisible by curve basis {curve.tenor_basis}")
                
            # If this is the first instrument being bootstrapped
            if len(curve.forward_rates) == 0:
                # Simplification: assume constant forward rate across all periods
                for i in range(num_periods):
                    start = i * curve.tenor_basis
                    curve.add_forward_rate(start, self.rate)
            else:
                # Calculate the forward rate for the last period
                # Requires prior periods to be already bootstrapped
                last_start = (num_periods - 1) * curve.tenor_basis
                
                # Check if we have all required prior forward rates
                for i in range(num_periods - 1):
                    start = i * curve.tenor_basis
                    if start not in curve.forward_rates:
                        raise ValueError(f"Missing forward rate at {start} needed for bootstrapping")
                
                # Calculate product of (1 + r_i * dt) for all prior periods
                compound_factor = 1.0
                for i in range(num_periods - 1):
                    start = i * curve.tenor_basis
                    fwd = curve.forward_rates[start]
                    compound_factor *= (1.0 + fwd * curve.tenor_basis)
                
                # Solve for the last forward rate
                # (1 + r_0*dt_0) * ... * (1 + r_n*dt_n) = 1/df
                last_forward = ((1.0 / df) / compound_factor - 1.0) / curve.tenor_basis
                curve.add_forward_rate(last_start, last_forward)
    
    def __repr__(self) -> str:
        return f"{self.index_name}Deposit(T={self.maturity:.2f}, r={self.rate*100:.4f}%)"


# In[14]:


@dataclass
class SOFRFuture:
    """SOFR Future contract"""
    start_date: float
    end_date: float
    price: float  # Future price (not rate)
    index_name: str = "SOFR"
    
    def bootstrap(self, forward_curve: ForwardCurve, discount_curve=None) -> None:
        """
        Bootstrap SOFR Future into forward curve
        Converts Future price to implied rate and calculates forward rate
        """
        # Check that tenor matches the curve's basis
        tenor = self.end_date - self.start_date
        if abs(tenor - forward_curve.tenor_basis) > 1e-6:
            raise ValueError(f"Future tenor {tenor} doesn't match curve basis {forward_curve.tenor_basis}")
        
        # Convert future price to implied rate (SOFR futures are quoted as 100 - rate)
        implied_rate = (100.0 - self.price) / 100.0
        
        # Apply convexity adjustment if discount curve is available
        if discount_curve is not None:
            # Simple convexity adjustment model
            # This is a simplified adjustment - a more sophisticated model would be needed in practice
            volatility = 0.002  # Assumed interest rate volatility
            time_to_maturity = self.start_date  # Time to future's start
            
            # Calculate discount factors
            df_start = discount_curve.get_discount_factor(self.start_date)
            df_end = discount_curve.get_discount_factor(self.end_date)
            
            # Simple convexity adjustment formula
            convexity_adjustment = volatility**2 * time_to_maturity * tenor
            
            # Apply adjustment to the implied rate
            forward_rate = implied_rate - convexity_adjustment
        else:
            # Without discount curve, use implied rate directly
            forward_rate = implied_rate
        
        # Add the forward rate to the curve
        forward_curve.add_forward_rate(self.start_date, forward_rate)
    
    def __repr__(self) -> str:
        implied_rate = (100.0 - self.price)
        return f"{self.index_name}Future({self.start_date:.2f}-{self.end_date:.2f}, price={self.price:.4f}, implied_rate={implied_rate:.4f}%)"


# In[15]:


@dataclass
class IBORSwap:
    """IBOR or SOFR Interest Rate Swap"""
    maturity: float
    rate: float
    index_name: str = "SOFR"
    fixed_frequency: int = 2  # Semi-annual fixed leg by default
    float_frequency: int = 4  # Quarterly floating leg by default
    
    def bootstrap(self, forward_curve: ForwardCurve, discount_curve: DiscountCurve) -> None:
        """
        Bootstrap interest rate swap into forward curve
        Uses multi-curve approach (OIS for discounting, separate curve for forwards)
        """
        # Generate payment schedules
        fixed_times = []
        fixed_interval = 1.0 / self.fixed_frequency
        current_time = fixed_interval
        while current_time <= self.maturity + 1e-10:
            fixed_times.append(current_time)
            current_time += fixed_interval
            
        float_times = []
        float_interval = 1.0 / self.float_frequency
        current_time = float_interval
        while current_time <= self.maturity + 1e-10:
            float_times.append(current_time)
            current_time += float_interval
            
        # Check tenor basis
        if abs(float_interval - forward_curve.tenor_basis) > 1e-10:
            raise ValueError(f"Swap floating frequency {float_interval} doesn't match curve basis {forward_curve.tenor_basis}")
        
        # Last floating period
        last_float_time = float_times[-1]
        last_float_start = last_float_time - float_interval
        
        # Check if already bootstrapped
        if last_float_start in forward_curve.forward_rates:
            return
            
        # Generate fixed leg cash flows
        fixed_cash_flows = []
        prev_time = 0.0
        for t in fixed_times:
            period = t - prev_time
            coupon = self.rate * period
            fixed_cash_flows.append(coupon)
            prev_time = t
            
        # PV of fixed leg
        fixed_pv = 0.0
        for t, cf in zip(fixed_times, fixed_cash_flows):
            df = discount_curve.get_discount_factor(t)
            fixed_pv += cf * df
            
        # PV of known floating leg payments
        float_pv = 0.0
        prev_time = 0.0
        
        for i, t in enumerate(float_times[:-1]):  # Exclude last payment
            start = prev_time
            end = t
            period = end - start
            
            # Get forward rate
            try:
                fwd_rate = forward_curve.get_forward_rate(start, end)
            except ValueError:
                raise ValueError(f"Missing forward rate for period [{start}, {end}]")
                
            # PV of this payment
            df = discount_curve.get_discount_factor(t)
            float_pv += fwd_rate * period * df
            
            prev_time = t
            
        # Solve for the last forward rate
        last_period = last_float_time - last_float_start
        last_df = discount_curve.get_discount_factor(last_float_time)
        
        # For par swap: fixed_pv = float_pv
        # => fixed_pv = known_float_pv + last_fwd * last_period * last_df
        # => last_fwd = (fixed_pv - known_float_pv) / (last_period * last_df)
        
        last_fwd_rate = (fixed_pv - float_pv) / (last_period * last_df)
        
        # Add to forward curve
        forward_curve.add_forward_rate(last_float_start, last_fwd_rate)
    
    def __repr__(self) -> str:
        fixed_freq = {1: "annual", 2: "semi-annual", 4: "quarterly"}.get(
            self.fixed_frequency, f"{self.fixed_frequency}/year")
        float_freq = {1: "annual", 2: "semi-annual", 4: "quarterly"}.get(
            self.float_frequency, f"{self.float_frequency}/year")
        return f"{self.index_name}Swap(T={self.maturity:.2f}, r={self.rate*100:.4f}%, {fixed_freq}/{float_freq})"


# In[17]:


get_ipython().system('jupyter nbconvert --to script forward_curve.ipynb')

