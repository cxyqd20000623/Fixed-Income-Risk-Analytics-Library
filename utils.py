#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import math
from scipy.optimize import minimize, newton
from scipy.interpolate import PchipInterpolator
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Union, Callable
from copy import deepcopy

# All utility classes: MathUtils, Interpolator, LogLinearInterpolator, 
# MonotonicCubicInterpolator, NelsonSiegelModel


# In[10]:


class MathUtils:
    """
    Utility class for financial math functions used across the framework
    """
    
    @staticmethod
    def calculate_zero_rate_from_df(df: float, tenor: float):
        """
        Calculate continuously compounded zero rate from discount factor
        
        Args:
            df: Discount factor
            tenor: Time to maturity in years
            
        Returns:
            Continuously compounded zero rate
        """
        if tenor <= 0 or df <= 0:
            return 0.0
        return -np.log(df) / tenor
    
    @staticmethod
    def calculate_df_from_zero_rate(rate: float, tenor: float):
        """
        Calculate discount factor from continuously compounded zero rate
        
        Args:
            rate: Zero rate
            tenor: Time to maturity in years
            
        Returns:
            Discount factor
        """
        if tenor <= 0:
            return 1.0
        return np.exp(-rate * tenor)
    
    @staticmethod
    def calculate_forward_rate(df_start: float, df_end: float, start_tenor: float, end_tenor: float):
        """
        Calculate simple forward rate from discount factors
        
        Args:
            df_start: Discount factor at start tenor
            df_end: Discount factor at end tenor
            start_tenor: Start tenor in years
            end_tenor: End tenor in years
            
        Returns:
            Simple forward rate for the period
        """
        if end_tenor <= start_tenor:
            raise ValueError("End tenor must be greater than start tenor")
        return (df_start / df_end - 1) / (end_tenor - start_tenor)
    
    @staticmethod
    def convert_ytm_to_zero_rate(ytm: float, frequency: int = 2):
        """
        Convert yield-to-maturity to continuous zero rate
        
        Args:
            ytm: Yield to maturity with compounding frequency per year
            frequency: Compounding frequency per year
            
        Returns:
            Equivalent continuously compounded zero rate
        """
        return frequency * np.log(1 + ytm / frequency)
    
    @staticmethod
    def convert_zero_rate_to_ytm(zero_rate: float, frequency: int = 2):
        """
        Convert continuous zero rate to yield-to-maturity
        
        Args:
            zero_rate: Continuously compounded zero rate
            frequency: Target compounding frequency per year
            
        Returns:
            Equivalent yield to maturity with specified compounding
        """
        return frequency * (np.exp(zero_rate / frequency) - 1)
    
    @staticmethod
    def solve_for_df(times: List[float], 
                     cash_flows: List[float], 
                     known_dfs: Dict[float, float],
                     target_price: float,
                     initial_guess: float = None):
        """
        Solve for the missing discount factor that makes PV equal to target price
        
        Args:
            times: Cash flow times
            cash_flows: Cash flow amounts
            known_dfs: Dictionary of known discount factors {time: df}
            target_price: Target present value
            initial_guess: Initial guess for the unknown discount factor
            
        Returns:
            The discount factor that makes the present value equal to target_price
        """
        if not times or not cash_flows:
            raise ValueError("Empty cash flows")
            
        final_time = times[-1]
        
        # Function to solve: PV(cash_flows) = target_price
        def equation(final_df):
            pv = 0.0
            for t, cf in zip(times, cash_flows):
                if t < final_time:
                    # Use known discount factors for earlier cash flows
                    if t in known_dfs:
                        df = known_dfs[t]
                    else:
                        # Interpolate if necessary
                        raise ValueError(f"Missing discount factor for time {t}")
                else:
                    # Use the unknown final discount factor
                    df = final_df
                    
                pv += cf * df
                
            return pv - target_price
        
        # Default initial guess if none provided
        if initial_guess is None:
            # Try to extrapolate from last two known points
            keys = sorted(known_dfs.keys())
            if len(keys) >= 2:
                t1, t2 = keys[-2], keys[-1]
                df1, df2 = known_dfs[t1], known_dfs[t2]
                # Log-linear extrapolation
                rate = -np.log(df2/df1) / (t2 - t1)
                initial_guess = df2 * np.exp(-rate * (final_time - t2))
            else:
                # Simple approximation
                initial_guess = 0.9 ** final_time
                
        # Use Newton's method to solve
        try:
            return newton(equation, initial_guess, tol=1e-10, maxiter=50)
        except:
            # Fallback to bisection
            a, b = 0.001, 0.999
            for _ in range(100):
                mid = (a + b) / 2
                if abs(equation(mid)) < 1e-10:
                    return mid
                if equation(mid) * equation(a) < 0:
                    b = mid
                else:
                    a = mid
            return (a + b) / 2
    
    @staticmethod
    def calculate_swap_rate(fixed_times: List[float],
                            float_times: List[float],
                            discount_factors: Dict[float, float],
                            float_rates: Dict[Tuple[float, float], float] = None,
                            float_period: float = 0.25):
        """
        Calculate par swap rate given discount factors and forward rates
        
        Args:
            fixed_times: Fixed leg payment times
            float_times: Floating leg payment times
            discount_factors: Dictionary of discount factors {time: df}
            float_rates: Dictionary of floating rates {(start, end): rate}
            float_period: Floating period length in years
            
        Returns:
            Par swap rate
        """
        # Calculate fixed leg annuity (sum of discounted period lengths)
        fixed_leg_annuity = 0.0
        prev_time = 0.0
        for t in fixed_times:
            period = t - prev_time
            if t in discount_factors:
                df = discount_factors[t]
            else:
                raise ValueError(f"Missing discount factor for time {t}")
                
            fixed_leg_annuity += period * df
            prev_time = t
            
        # If no floating rates provided, assume par valuation
        if float_rates is None:
            # For a par swap, floating leg PV = 1 - DF(maturity)
            float_leg_pv = 1.0 - discount_factors[float_times[-1]]
        else:
            # Calculate floating leg PV from forward rates
            float_leg_pv = 0.0
            prev_time = 0.0
            for t in float_times:
                period = t - prev_time
                if (prev_time, t) in float_rates:
                    rate = float_rates[(prev_time, t)]
                else:
                    raise ValueError(f"Missing forward rate for period [{prev_time}, {t}]")
                    
                if t in discount_factors:
                    df = discount_factors[t]
                else:
                    raise ValueError(f"Missing discount factor for time {t}")
                    
                float_leg_pv += rate * period * df
                prev_time = t
                
        # Par swap rate = Floating leg PV / Fixed leg annuity
        return float_leg_pv / fixed_leg_annuity
    
    @staticmethod
    def create_bumped_curve(original_curve, bump_size):
        """Create a parallel-shifted version of any curve type"""
        if original_curve is None:
            return None
            
        # Create a deep copy of the curve
        from copy import deepcopy
        bumped_curve = deepcopy(original_curve)
        
        # If it's a ZeroCurve
        if hasattr(bumped_curve, 'zero_rates'):
            for tenor in bumped_curve.zero_rates:
                bumped_curve.zero_rates[tenor] += bump_size
                
        # If it's a discount curve
        elif hasattr(bumped_curve, 'discount_factors'):
            for tenor, df in list(bumped_curve.discount_factors.items()):
                if tenor > 0:
                    zero_rate = -np.log(df) / tenor
                    bumped_zero = zero_rate + bump_size
                    bumped_df = np.exp(-bumped_zero * tenor)
                    bumped_curve.discount_factors[tenor] = bumped_df
                    
        # If it's a forward curve
        elif hasattr(bumped_curve, 'forward_rates'):
            for tenor in bumped_curve.forward_rates:
                bumped_curve.forward_rates[tenor] += bump_size
                
        return bumped_curve


# In[11]:


class Interpolator(ABC):
    """Abstract base class for interpolation methods"""
    
    @abstractmethod
    def interpolate(self, x: float, x_values: List[float], y_values: List[float]):
        """Interpolate at point x given points (x_values, y_values)"""
        pass


# In[16]:


class LogLinearInterpolator(Interpolator):
    """
    Log-linear interpolation - appropriate for discount factors
    Preserves exponential decay characteristic
    """

    def interpolate(self, x: float, x_values: List[float], y_values: List[float]):
        """
        Perform log-linear interpolation at point x with extrapolation capability

        Args:
            x: The point to interpolate at
            x_values: Array of x coordinates
            y_values: Array of y coordinates (must be positive)

        Returns:
            Interpolated value at x
        """
        if not x_values or not y_values:
            raise ValueError("Empty data points for interpolation")

        # Handle extrapolation
        if x < min(x_values):
            # Extrapolate for short end - use first value
            return y_values[0]
        elif x > max(x_values):
            # Extrapolate for long end - use log-linear extrapolation from last two points
            if len(x_values) >= 2:
                x1, x2 = x_values[-2], x_values[-1]
                y1, y2 = y_values[-2], y_values[-1]

                # Check if values are positive (required for log-linear)
                if y1 <= 0 or y2 <= 0:
                    # Use linear extrapolation instead
                    slope = (y2 - y1) / (x2 - x1)
                    return y2 + slope * (x - x2)

                # Use log-linear extrapolation
                rate = np.log(y2 / y1) / (x2 - x1)
                return y2 * np.exp(rate * (x - x2))
            else:
                # Only one point available, assume flat
                return y_values[-1]

        # Regular interpolation (x is within range)
        # Find bracketing indices
        idx = None
        for i in range(len(x_values) - 1):
            if x_values[i] <= x <= x_values[i+1]:
                idx = i
                break

        if idx is None:
            raise ValueError(f"Cannot find bracketing interval for {x}")

        x1, x2 = x_values[idx], x_values[idx+1]
        y1, y2 = y_values[idx], y_values[idx+1]

        # Handle edge cases
        if abs(x2 - x1) < 1e-10:
            return y1
        if y1 <= 0 or y2 <= 0:
            # Fall back to linear for negative or zero values
            alpha = (x - x1) / (x2 - x1)
            return y1 + alpha * (y2 - y1)

        # Log-linear interpolation
        alpha = (x - x1) / (x2 - x1)
        return y1 * (y2/y1)**alpha


# In[13]:


class MonotonicCubicInterpolator(Interpolator):
    """
    Monotonic cubic spline interpolation (PCHIP)
    Preserves monotonicity and shape characteristics
    """
    
    def __init__(self):
        self.pchip = None
        self.x_min = None
        self.x_max = None
        
    def fit(self, x_values: List[float], y_values: List[float]):
        """
        Fit the interpolator to data points
        
        Args:
            x_values: X coordinates (must be increasing)
            y_values: Y coordinates
        """
        if len(x_values) < 2:
            raise ValueError("At least two points required for interpolation")
            
        # Ensure x values are increasing
        if not all(x_values[i] < x_values[i+1] for i in range(len(x_values)-1)):
            # Sort points by x
            indices = np.argsort(x_values)
            x_values = [x_values[i] for i in indices]
            y_values = [y_values[i] for i in indices]
            
        self.pchip = PchipInterpolator(x_values, y_values)
        self.x_min = min(x_values)
        self.x_max = max(x_values)
        
    def interpolate(self, x: float, x_values: List[float] = None, y_values: List[float] = None):
        """
        Interpolate at point x
        
        Args:
            x: Point to interpolate at
            x_values: X coordinates (optional if already fit)
            y_values: Y coordinates (optional if already fit)
            
        Returns:
            Interpolated value
        """
        # Fit if not already fit or new data provided
        if (self.pchip is None or x_values is not None) and y_values is not None:
            self.fit(x_values, y_values)
            
        if self.pchip is None:
            raise ValueError("Interpolator not fit to data")
            
        if x < self.x_min or x > self.x_max:
            raise ValueError(f"Point {x} outside range [{self.x_min}, {self.x_max}]")
            
        return float(self.pchip(x))


# In[14]:


class NelsonSiegelModel:
    """
    Nelson-Siegel model for yield curve smoothing
    
    r(t) = β₀ + β₁[(1-e^(-t/τ))/(t/τ)] + β₂[(1-e^(-t/τ))/(t/τ) - e^(-t/τ)]
    """
    
    def __init__(self):
        self.beta0 = 0.0  # Long-term level
        self.beta1 = 0.0  # Short-term component
        self.beta2 = 0.0  # Medium-term component
        self.tau = 1.0    # Decay factor
        self.is_calibrated = False
        
    def calibrate(self, tenors, rates):
        """Calibrate model parameters to market rates"""
        # Filter out very short tenors to avoid numerical issues
        filtered_tenors = []
        filtered_rates = []
        for t, r in zip(tenors, rates):
            if t > 1e-6:  # Exclude near-zero tenors
                filtered_tenors.append(t)
                filtered_rates.append(r)
        
        if len(filtered_tenors) < 4:
            raise ValueError("Need at least 4 data points to calibrate Nelson-Siegel model")
        
        # Initial guess: long rate, short-long difference, curvature, decay
        initial_guess = [
            filtered_rates[-1],                    # β₀ = longest tenor rate
            filtered_rates[0] - filtered_rates[-1], # β₁ = short rate - long rate
            0.0,                                   # β₂ = initial guess of zero
            max(2.0, filtered_tenors[-1] / 4)       # τ = reasonable starting value
        ]
        
        # Define objective function: sum of squared errors
        def objective(params):
            beta0, beta1, beta2, tau = params
            errors = []
            for t, r in zip(filtered_tenors, filtered_rates):
                model_rate = self._ns_formula(t, beta0, beta1, beta2, tau)
                errors.append((model_rate - r) ** 2)
            return np.sum(errors)
        
        # Parameter bounds to ensure economically meaningful results
        bounds = [
            (0.0, None),    # β₀ > 0 (positive long rate)
            (None, None),   # β₁ can be any value
            (None, None),   # β₂ can be any value
            (0.1, 10.0)     # 0.1 < τ < 10.0 (reasonable decay range)
        ]
        
        # Minimize the objective function
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        if not result.success:
            print(f"Warning: Nelson-Siegel calibration may not have converged: {result.message}")
            
        # Store calibrated parameters
        self.beta0, self.beta1, self.beta2, self.tau = result.x
        self.is_calibrated = True
        
        # Print calibration quality metrics
        mse = objective(result.x) / len(filtered_tenors)
        print(f"Nelson-Siegel calibration: MSE={mse:.8f}, params={result.x}")
        
    def get_value(self, t: float):
        """Get Nelson-Siegel rate at tenor t"""
        if not self.is_calibrated:
            raise RuntimeError("Model must be calibrated before use")
            
        return self._ns_formula(t, self.beta0, self.beta1, self.beta2, self.tau)
        
    def _ns_formula(self, t: float, beta0: float, beta1: float, beta2: float, tau: float):
        """Nelson-Siegel formula implementation"""
        if t < 1e-6:  # Handle very short tenors
            return beta0 + beta1
            
        exp_term = np.exp(-t / tau)
        term1 = (1 - exp_term) / (t / tau)
        term2 = term1 - exp_term
        
        return beta0 + beta1 * term1 + beta2 * term2


# In[17]:


get_ipython().system('jupyter nbconvert --to script utils.ipynb')


# In[ ]:




