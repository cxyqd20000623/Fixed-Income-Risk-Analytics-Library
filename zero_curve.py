#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from dataclasses import dataclass
from scipy.optimize import newton
from abstract_curve import InterestRateCurve
from utils import MathUtils, Interpolator, MonotonicCubicInterpolator, NelsonSiegelModel

# ZeroCurve class
# TreasuryBond class


# In[3]:


class ZeroCurve(InterestRateCurve):
    """
    Zero rate curve for a specific market (e.g., Treasury)
    Stores zero rates and calculates implied discount factors
    """
    
    def __init__(self, interpolator: Interpolator = None, model: NelsonSiegelModel = None):
        self.zero_rates = {}  # {tenor: zero_rate}
        self.interpolator = interpolator or MonotonicCubicInterpolator()
        self.interpolator_needs_update = True
        self.model = model
        self.use_model = False
        
    def add_zero_rate(self, tenor: float, rate: float) -> None:
        """Add zero rate to curve"""
        self.zero_rates[tenor] = rate
        self.interpolator_needs_update = True
        
    def _update_interpolator(self) -> None:
        """Update the cubic spline interpolator with current data"""
        if not self.interpolator_needs_update:
            return
            
        if not isinstance(self.interpolator, MonotonicCubicInterpolator):
            return
            
        tenors = sorted(self.zero_rates.keys())
        rates = [self.zero_rates[t] for t in tenors]
        
        if len(tenors) >= 2:
            self.interpolator.fit(tenors, rates)
            self.interpolator_needs_update = False
        
    def get_zero_rate(self, t: float) -> float:
        """Get continuously compounded zero rate"""
        if t <= 0:
            return 0.0
            
        if t in self.zero_rates:
            return self.zero_rates[t]
            
        if self.use_model and self.model is not None:
            return self.model.get_value(t)
            
        # Use interpolation
        try:
            if isinstance(self.interpolator, MonotonicCubicInterpolator):
                self._update_interpolator()
                return self.interpolator.interpolate(t)
            else:
                tenors = sorted(self.zero_rates.keys())
                rates = [self.zero_rates[t] for t in tenors]
                return self.interpolator.interpolate(t, tenors, rates)
                
        except ValueError as e:
            # Handle extrapolation if needed
            tenors = sorted(self.zero_rates.keys())
            
            if t < tenors[0]:
                # Simple extrapolation for short end
                return self.zero_rates[tenors[0]]
            elif t > tenors[-1]:
                # Simple extrapolation for long end
                return self.zero_rates[tenors[-1]]
            else:
                raise ValueError(f"Cannot interpolate zero rate at {t}: {e}")
        
    def get_discount_factor(self, t: float) -> float:
        """Calculate discount factor from zero rate"""
        if t <= 0:
            return 1.0
        r = self.get_zero_rate(t)
        return MathUtils.calculate_df_from_zero_rate(r, t)
        
    def apply_model(self) -> None:
        """Apply Nelson-Siegel model for smoothing"""
        if self.model is None:
            self.model = NelsonSiegelModel()
            
        tenors = sorted(self.zero_rates.keys())
        rates = [self.zero_rates[t] for t in tenors]
        
        self.model.calibrate(tenors, rates)
        self.use_model = True
        
    def _get_max_tenor(self) -> float:
        """Get maximum tenor in the curve"""
        if self.zero_rates:
            return max(self.zero_rates.keys())
        return 10.0
        
    def __repr__(self) -> str:
        lines = ["ZeroCurve:"]
        for tenor in sorted(self.zero_rates.keys()):
            zr = self.zero_rates[tenor] * 100  # To percent
            df = self.get_discount_factor(tenor)
            lines.append(f"  T={tenor:.2f} -> ZR={zr:.4f}%, DF={df:.8f}")
        if self.use_model:
            lines.append("  (Nelson-Siegel model applied)")
        return "\n".join(lines)


# In[4]:


@dataclass
class TreasuryBond:
    """US Treasury Bond or Note"""
    maturity: float
    ytm: float  # Yield to maturity (semi-annual compounding)
    coupon_rate: float = 0.0  # Annual coupon rate (paid semi-annually)
    
    def bootstrap(self, curve: ZeroCurve) -> None:
        """
        Bootstrap Treasury bond into zero curve
        Extracts zero rate that replicates the bond's market price
        """
        if self.coupon_rate == 0:
            # Zero coupon bond - direct conversion from YTM to zero rate
            zero_rate = MathUtils.convert_ytm_to_zero_rate(self.ytm, frequency=2)
            curve.add_zero_rate(self.maturity, zero_rate)
            return
        
        # Generate cash flow schedule
        times = []
        cash_flows = []
        
        # Semi-annual coupons
        current_time = 0.5
        while current_time <= self.maturity + 1e-10:
            times.append(current_time)
            payment = self.coupon_rate / 2  # Semi-annual coupon
            if abs(current_time - self.maturity) < 1e-10:
                payment += 1.0  # Add principal at maturity
            cash_flows.append(payment)
            current_time += 0.5
            
        # Calculate bond price using YTM
        bond_price = 0.0
        for t, cf in zip(times, cash_flows):
            df = 1.0 / ((1 + self.ytm/2) ** (t * 2))
            bond_price += cf * df
            
        # Create a temporary bond object to extract zero rate
        temp_bond = Bond(
            maturity=self.maturity,
            coupon_rate=self.coupon_rate,
            frequency=2,
            face_value=1.0
        )
        
        # Function to solve: find zero rate that gives YTM price
        def equation(zero_rate):
            # Create temporary discount factors
            temp_dfs = {t: np.exp(-zero_rate * t) for t in times}
            
            # Calculate price with these discount factors
            price = sum(cf * temp_dfs[t] for t, cf in zip(times, cash_flows))
            
            return price - bond_price
            
        # Initial guess - use converted YTM
        initial_guess = MathUtils.convert_ytm_to_zero_rate(self.ytm, frequency=2)
        
        # Solve for zero rate using Newton's method
        try:
            zero_rate = newton(equation, initial_guess, tol=1e-10, maxiter=50)
        except:
            # Fallback to bisection
            a, b = max(0.0, initial_guess - 0.05), initial_guess + 0.05
            for _ in range(100):
                mid = (a + b) / 2
                if abs(equation(mid)) < 1e-10:
                    zero_rate = mid
                    break
                if equation(mid) * equation(a) < 0:
                    b = mid
                else:
                    a = mid
            else:
                zero_rate = (a + b) / 2
                
        # Add to curve
        curve.add_zero_rate(self.maturity, zero_rate)
    
    def __repr__(self) -> str:
        if self.coupon_rate == 0:
            return f"TreasuryBill(T={self.maturity:.2f}, YTM={self.ytm*100:.4f}%)"
        else:
            return f"TreasuryBond(T={self.maturity:.2f}, coupon={self.coupon_rate*100:.2f}%, YTM={self.ytm*100:.4f}%)"


# In[ ]:


get_ipython().system('jupyter nbconvert --to script zero_curve.ipynb')

