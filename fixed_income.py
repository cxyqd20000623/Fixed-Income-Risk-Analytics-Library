#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from utils import MathUtils
from abstract_curve import InterestRateCurve
from discount_curve import DiscountCurve
from forward_curve import ForwardCurve
from zero_curve import ZeroCurve

# FixedIncomeInstrument abstract class
# Bond class
# InterestRateSwap class


# In[4]:


class FixedIncomeInstrument(ABC):
    """
    Abstract base class for fixed income instruments
    Defines common interface for pricing and risk measures
    """
    
    @abstractmethod
    def price(self, valuation_curve=None):
        """
        Calculate the present value of the instrument
        
        Args:
            valuation_curve: Curve to use for valuation (defaults to instrument's curve)
            
        Returns:
            Present value of the instrument
        """
        pass
    
    @abstractmethod
    def dv01(self, bump_size=0.0001):
        """
        Calculate DV01 (dollar value of 1 basis point change)
        
        Args:
            bump_size: Size of the yield bump in decimal (default 0.0001 = 1bp)
            
        Returns:
            DV01 value (per 100 face value/notional)
        """
        pass
    
    def pv01(self, bump_size=0.0001):
        """
        Calculate PV01 (present value of 1 basis point change)
        
        Args:
            bump_size: Size of the yield bump in decimal
            
        Returns:
            PV01 value
        """
        # Default implementation - PV01 is typically -DV01
        return -self.dv01(bump_size)
    
    @abstractmethod
    def duration(self):
        """
        Calculate duration of the instrument
        
        Returns:
            Duration in years
        """
        pass
    
    @abstractmethod
    def convexity(self):
        """
        Calculate convexity of the instrument
        
        Returns:
            Convexity measure
        """
        pass
    
    def risk_report(self):
        """
        Generate standardized risk report
        
        Returns:
            Dictionary of risk measures
        """
        return {
            "price": self.price(),
            "duration": self.duration(),
            "modified_duration": getattr(self, "modified_duration", lambda: self.duration())(),
            "convexity": self.convexity(),
            "dv01": self.dv01(),
            "pv01": self.pv01()
        }


# In[5]:


class Bond(FixedIncomeInstrument):
    """
    Fixed income bond pricing and risk measures
    Supports Treasury bonds, corporate bonds, and other fixed-rate debt instruments
    """
    
    def __init__(self, 
                maturity: float, 
                coupon_rate: float, 
                frequency: int = 2, 
                discount_curve=None, 
                spread: float = 0.0,
                face_value: float = 100.0):
        """
        Initialize a bond
        
        Args:
            maturity: Time to maturity in years
            coupon_rate: Annual coupon rate (e.g., 0.05 for 5%)
            frequency: Coupon payments per year (default 2 = semi-annual)
            discount_curve: Curve used for discounting (ZeroCurve or DiscountCurve)
            spread: Spread over the base curve in decimal (e.g., 0.01 for 100bps)
            face_value: Face value of the bond (default 100)
        """
        self.maturity = maturity
        self.coupon_rate = coupon_rate
        self.frequency = frequency
        self.discount_curve = discount_curve
        self.spread = spread
        self.face_value = face_value
        
        # Create payment schedule
        self.payment_times = []
        self.payment_amounts = []
        
        payment_interval = 1.0 / frequency
        current_time = payment_interval
        
        # Add regular coupon payments
        while current_time <= maturity + 1e-10:
            self.payment_times.append(current_time)
            self.payment_amounts.append(coupon_rate * face_value / frequency)
            current_time += payment_interval
            
        # Add principal repayment to the final payment
        if self.payment_times:
            self.payment_amounts[-1] += face_value
    
    def price(self, valuation_curve=None):
        """
        Calculate the present value of the bond
        
        Args:
            valuation_curve: Curve to use for discounting (defaults to self.discount_curve)
            
        Returns:
            Present value of the bond
        """
        if valuation_curve is None:
            valuation_curve = self.discount_curve
            
        if valuation_curve is None:
            raise ValueError("No valuation curve provided")
            
        # Sum present values of all cash flows
        bond_price = 0.0
        for t, amount in zip(self.payment_times, self.payment_amounts):
            # Apply spread if specified
            if self.spread != 0:
                # Adjust discount factor by spread
                df = valuation_curve.get_discount_factor(t) * np.exp(-self.spread * t)
            else:
                df = valuation_curve.get_discount_factor(t)
                
            bond_price += amount * df
            
        return bond_price
    
    def yield_to_maturity(self, market_price=None):
        """
        Calculate the yield to maturity for a given market price
        
        Args:
            market_price: Market price of the bond (if None, uses calculated price)
            
        Returns:
            Yield to maturity (annualized)
        """
        if market_price is None:
            market_price = self.price()
            
        # Use the same algorithm as TreasuryBond.bootstrap()
        def price_function(ytm):
            price = 0.0
            for t, amount in zip(self.payment_times, self.payment_amounts):
                # Discount factor using periodic compounding
                df = 1.0 / ((1.0 + ytm/self.frequency) ** (t * self.frequency))
                price += amount * df
            return price
        
        def equation(ytm):
            return price_function(ytm) - market_price
        
        # Initial guess - use simple approximation
        annual_coupon = self.coupon_rate * self.face_value
        principal = self.face_value
        simple_yield = (annual_coupon + (principal - market_price) / self.maturity) / ((principal + market_price) / 2)
        initial_guess = max(0.001, simple_yield)
        
        # Use Newton's method to solve for YTM
        try:
            return newton(equation, initial_guess, tol=1e-10, maxiter=50)
        except:
            # Fall back to bisection
            a, b = 0.0, 1.0
            while price_function(b) > market_price:
                b *= 2
                
            for _ in range(100):
                mid = (a + b) / 2
                price_mid = price_function(mid)
                if abs(price_mid - market_price) < 1e-10:
                    return mid
                if price_mid > market_price:
                    a = mid
                else:
                    b = mid
            return (a + b) / 2
    
    def duration(self, market_price=None):
        """
        Calculate Macaulay duration of the bond
        
        Args:
            market_price: Market price (if None, uses calculated price)
            
        Returns:
            Macaulay duration in years
        """
        if market_price is None:
            market_price = self.price()
            
        ytm = self.yield_to_maturity(market_price)
        
        # Calculate weighted average time
        duration = 0.0
        for t, amount in zip(self.payment_times, self.payment_amounts):
            df = 1.0 / ((1.0 + ytm/self.frequency) ** (t * self.frequency))
            duration += t * amount * df / market_price
            
        return duration
    
    def modified_duration(self, market_price=None):
        """
        Calculate modified duration (price sensitivity to yield)
        
        Args:
            market_price: Market price (if None, uses calculated price)
            
        Returns:
            Modified duration in years
        """
        if market_price is None:
            market_price = self.price()
            
        ytm = self.yield_to_maturity(market_price)
        mac_duration = self.duration(market_price)
        
        # Modified duration = Macaulay duration / (1 + ytm/frequency)
        return mac_duration / (1.0 + ytm/self.frequency)
    
    def convexity(self, market_price=None):
        """
        Calculate convexity of the bond
        
        Args:
            market_price: Market price (if None, uses calculated price)
            
        Returns:
            Convexity measure
        """
        if market_price is None:
            market_price = self.price()
            
        ytm = self.yield_to_maturity(market_price)
        
        # Calculate convexity (second derivative)
        convexity = 0.0
        for t, amount in zip(self.payment_times, self.payment_amounts):
            df = 1.0 / ((1.0 + ytm/self.frequency) ** (t * self.frequency))
            # t * (t + 1/frequency) is an approximation for d²PV/dy²
            convexity += t * (t + 1.0/self.frequency) * amount * df / market_price
            
        # Normalize by (1 + ytm/frequency)²
        return convexity / (1.0 + ytm/self.frequency)**2
    
    def dv01(self, bump_size=0.0001):
        """
        Calculate DV01 (dollar value of 1 basis point change)
        
        Args:
            bump_size: Size of the yield bump in decimal (default 0.0001 = 1bp)
            
        Returns:
            DV01 value (per 100 face value)
        """
        # Base price
        base_price = self.price()
        
        # Create bumped discount curve
        if self.discount_curve is not None:
            # Use the utility method instead of the local method
            bumped_curve = MathUtils.create_bumped_curve(self.discount_curve, bump_size)
            bumped_price = self.price(bumped_curve)
        else:
            # Without a curve, use YTM approach
            ytm = self.yield_to_maturity(base_price)
            
            # Bumped price
            price_function = lambda y: sum(amount / ((1 + y/self.frequency) ** (t * self.frequency)) 
                                  for t, amount in zip(self.payment_times, self.payment_amounts))
            bumped_price = price_function(ytm + bump_size)
        
        # DV01 = change in price for 1bp move (scaled to 100 face value)
        return (base_price - bumped_price) * 100 / self.face_value
        
    def __repr__(self):
        if self.coupon_rate == 0:
            return f"ZeroCouponBond(maturity={self.maturity:.2f}, face_value={self.face_value:.2f})"
        else:
            return f"Bond(maturity={self.maturity:.2f}, coupon={self.coupon_rate*100:.2f}%, frequency={self.frequency}/year)"


# In[6]:


class InterestRateSwap(FixedIncomeInstrument):
    """
    Interest rate swap pricing and risk measures
    Supports fixed-for-floating swaps using the multi-curve framework
    """
    
    def __init__(self,
                maturity: float,
                fixed_rate: float,
                index_name: str = "SOFR",
                notional: float = 1000000.0,
                fixed_frequency: int = 2,
                float_frequency: int = 4,
                discount_curve=None,
                forward_curve=None,
                receive_fixed: bool = True):
        """
        Initialize an interest rate swap
        
        Args:
            maturity: Time to maturity in years
            fixed_rate: Fixed leg rate (annual, e.g., 0.05 for 5%)
            index_name: Floating index name (e.g., "SOFR", "LIBOR")
            notional: Notional amount
            fixed_frequency: Fixed leg payments per year (default 2 = semi-annual)
            float_frequency: Floating leg payments per year (default 4 = quarterly)
            discount_curve: Curve used for discounting (typically OIS)
            forward_curve: Curve used for projecting floating rates
            receive_fixed: True if receiving fixed/paying floating, False otherwise
        """
        self.maturity = maturity
        self.fixed_rate = fixed_rate
        self.index_name = index_name
        self.notional = notional
        self.fixed_frequency = fixed_frequency
        self.float_frequency = float_frequency
        self.discount_curve = discount_curve
        self.forward_curve = forward_curve
        self.receive_fixed = receive_fixed
        
        # Create fixed leg payment schedule
        self.fixed_times = []
        fixed_interval = 1.0 / fixed_frequency
        current_time = fixed_interval
        
        while current_time <= maturity + 1e-10:
            self.fixed_times.append(current_time)
            current_time += fixed_interval
        
        # Create floating leg payment schedule
        self.float_times = []
        float_interval = 1.0 / float_frequency
        current_time = float_interval
        
        while current_time <= maturity + 1e-10:
            self.float_times.append(current_time)
            current_time += float_interval
    
    def fixed_leg_pv(self, valuation_discount_curve=None):
        """
        Calculate present value of fixed leg
        
        Args:
            valuation_discount_curve: Curve for discounting
            
        Returns:
            Present value of fixed leg
        """
        if valuation_discount_curve is None:
            valuation_discount_curve = self.discount_curve
            
        if valuation_discount_curve is None:
            raise ValueError("No discount curve provided")
            
        fixed_pv = 0.0
        prev_time = 0.0
        
        for t in self.fixed_times:
            # Calculate fixed coupon
            dt = t - prev_time
            coupon = self.fixed_rate * dt * self.notional
            
            # Discount to present value
            df = valuation_discount_curve.get_discount_factor(t)
            fixed_pv += coupon * df
            
            prev_time = t
            
        return fixed_pv
    
    def floating_leg_pv(self, valuation_discount_curve=None, valuation_forward_curve=None):
        """
        Calculate present value of floating leg
        
        Args:
            valuation_discount_curve: Curve for discounting
            valuation_forward_curve: Curve for projecting rates
            
        Returns:
            Present value of floating leg
        """
        if valuation_discount_curve is None:
            valuation_discount_curve = self.discount_curve
            
        if valuation_forward_curve is None:
            valuation_forward_curve = self.forward_curve
            
        if valuation_discount_curve is None or valuation_forward_curve is None:
            raise ValueError("Discount and forward curves required")
            
        float_pv = 0.0
        prev_time = 0.0
        
        for t in self.float_times:
            # Get forward rate for the period
            start = prev_time
            end = t
            dt = end - start
            
            try:
                fwd_rate = valuation_forward_curve.get_forward_rate(start, end)
            except:
                # Try with default tenor basis
                fwd_rate = valuation_forward_curve.get_forward_rate(start)
                
            # Calculate projected payment
            payment = fwd_rate * dt * self.notional
            
            # Discount to present value
            df = valuation_discount_curve.get_discount_factor(t)
            float_pv += payment * df
            
            prev_time = t
            
        return float_pv
    
    def price(self, valuation_discount_curve=None, valuation_forward_curve=None):
        """
        Calculate present value of the swap
        
        Args:
            valuation_discount_curve: Curve for discounting
            valuation_forward_curve: Curve for projecting rates
            
        Returns:
            Present value from perspective of fixed rate receiver
        """
        fixed_pv = self.fixed_leg_pv(valuation_discount_curve)
        float_pv = self.floating_leg_pv(valuation_discount_curve, valuation_forward_curve)
        
        # Value depends on direction
        if self.receive_fixed:
            return fixed_pv - float_pv
        else:
            return float_pv - fixed_pv
    
    def par_rate(self, valuation_discount_curve=None, valuation_forward_curve=None):
        """
        Calculate par swap rate
        
        Args:
            valuation_discount_curve: Curve for discounting
            valuation_forward_curve: Curve for projecting rates
            
        Returns:
            Par fixed rate for the swap
        """
        if valuation_discount_curve is None:
            valuation_discount_curve = self.discount_curve
            
        if valuation_forward_curve is None:
            valuation_forward_curve = self.forward_curve
            
        if valuation_discount_curve is None or valuation_forward_curve is None:
            raise ValueError("Discount and forward curves required")
            
        # Get discount factors for fixed times
        discount_factors = {t: valuation_discount_curve.get_discount_factor(t) for t in self.fixed_times}
        
        # Get forward rates for floating periods
        forward_rates = {}
        prev_time = 0.0
        for t in self.float_times:
            try:
                forward_rates[(prev_time, t)] = valuation_forward_curve.get_forward_rate(prev_time, t)
            except:
                # Try with default tenor basis
                forward_rates[(prev_time, t)] = valuation_forward_curve.get_forward_rate(prev_time)
            prev_time = t
            
        # Use MathUtils to calculate par swap rate
        return MathUtils.calculate_swap_rate(
            self.fixed_times,
            self.float_times,
            discount_factors,
            forward_rates)
    
    def duration(self):
        """
        Calculate effective duration of the swap
        
        Returns:
            Effective duration in years
        """
        return self.effective_duration()
        
    def effective_duration(self, bump_size=0.0001):
        """
        Calculate effective duration
        
        Args:
            bump_size: Size of the interest rate bump in decimal (default 0.0001 = 1bp)
            
        Returns:
            Effective duration in years
        """
        # Base price
        base_price = self.price()
        if abs(base_price) < 1e-10:
            return 0.0  # At par, duration is close to zero
            
        # Up bump (parallel shift up of both curves)
        up_discount = MathUtils.create_bumped_curve(self.discount_curve, bump_size)
        up_forward = MathUtils.create_bumped_curve(self.forward_curve, bump_size)
        up_price = self.price(up_discount, up_forward)
        
        # Down bump (parallel shift down of both curves)
        down_discount = MathUtils.create_bumped_curve(self.discount_curve, -bump_size)
        down_forward = MathUtils.create_bumped_curve(self.forward_curve, -bump_size)
        down_price = self.price(down_discount, down_forward)
        
        # Calculate effective duration
        # Duration = -1 * (P(+) - P(-)) / (2 * dY * P(0))
        if abs(base_price) < 1e-10:
            # Handle near-zero base price - use absolute change instead
            return -1.0 * (up_price - down_price) / (2 * bump_size * self.notional)
        else:
            return -1.0 * (up_price - down_price) / (2 * bump_size * base_price)
    
    def effective_convexity(self, bump_size=0.0001):
        """
        Calculate effective convexity
        
        Args:
            bump_size: Size of the yield bump in decimal
            
        Returns:
            Effective convexity
        """
        # Base price
        base_price = self.price()
        
        # Up bump (parallel shift up of both curves)
        up_discount = MathUtils.create_bumped_curve(self.discount_curve, bump_size)
        up_forward = MathUtils.create_bumped_curve(self.forward_curve, bump_size)
        up_price = self.price(up_discount, up_forward)
        
        # Down bump (parallel shift down of both curves)
        down_discount = MathUtils.create_bumped_curve(self.discount_curve, -bump_size)
        down_forward = MathUtils.create_bumped_curve(self.forward_curve, -bump_size)
        down_price = self.price(down_discount, down_forward)
        
        # Calculate effective convexity
        # Convexity = (P(+) + P(-) - 2*P(0)) / (dY^2 * P(0))
        if abs(base_price) < 1e-10:
            # Handle near-zero base price - use absolute change instead
            return (up_price + down_price - 2 * base_price) / (bump_size**2 * self.notional)
        else:
            return (up_price + down_price - 2 * base_price) / (bump_size**2 * base_price)
    
    def dv01(self, curve_type='both', bump_size=0.0001):
        """
        Calculate DV01 (dollar value of 1 basis point change)
        
        Args:
            curve_type: 'discount', 'forward', or 'both'
            bump_size: Size of yield bump in decimal
            
        Returns:
            DV01 value per 100 notional
        """
        # Base price
        base_price = self.price()
        
        if curve_type == 'discount':
            # Bump only discount curve
            bumped_curve = MathUtils.create_bumped_curve(self.discount_curve, bump_size)
            bumped_price = self.price(valuation_discount_curve=bumped_curve)
            
        elif curve_type == 'forward':
            # Bump only forward curve
            bumped_curve = MathUtils.create_bumped_curve(self.forward_curve, bump_size)
            bumped_price = self.price(valuation_forward_curve=bumped_curve)
            
        else:  # 'both'
            # Bump both curves
            disc_bumped = MathUtils.create_bumped_curve(self.discount_curve, bump_size)
            fwd_bumped = MathUtils.create_bumped_curve(self.forward_curve, bump_size)
            bumped_price = self.price(disc_bumped, fwd_bumped)
            
        # DV01 = change in price for 1bp move (scaled to 100 notional)
        return (base_price - bumped_price) * 100 / self.notional
    
    def convexity(self):
        """Calculate convexity of the swap"""
        return self.effective_convexity()
        
    def basis_point_value(self, curve_type='forward'):
        """
        Calculate basis point value (effect of 1bp change in floating rate index)
        
        Args:
            curve_type: 'forward' or 'discount'
            
        Returns:
            Basis point value per 100 notional
        """
        # This is essentially the same as dv01 for the specified curve
        return self.dv01(curve_type)
    
    def __repr__(self):
        direction = "Receive" if self.receive_fixed else "Pay"
        return f"{direction}-Fixed Swap(maturity={self.maturity:.2f}, fixed={self.fixed_rate*100:.2f}%, {self.index_name}, notional={self.notional:,.0f})"


# In[7]:


get_ipython().system('jupyter nbconvert --to script fixed_income.ipynb')

