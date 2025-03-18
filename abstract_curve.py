#!/usr/bin/env python
# coding: utf-8

# In[4]:


from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from utils import MathUtils


# In[5]:


class InterestRateCurve(ABC):
    """Abstract base class for all interest rate curves"""
    
    @abstractmethod
    def get_discount_factor(self, t: float):
        """Get discount factor for tenor t"""
        pass
        
    @abstractmethod
    def get_zero_rate(self, t: float) -> float:
        """Get continuously compounded zero rate for tenor t"""
        pass
        
    def get_forward_rate(self, start: float, end: float):
        """
        Get simple forward rate from start to end
        
        Default implementation using discount factors
        """
        if end <= start:
            raise ValueError("End time must be greater than start time")
            
        df_start = self.get_discount_factor(start)
        df_end = self.get_discount_factor(end)
        
        return MathUtils.calculate_forward_rate(df_start, df_end, start, end)
    
    def plot(self, title: str = "Interest Rate Curve", max_tenor: float = None):
        """Plot curve visualization with discount factors, zero rates, and forwards"""
        if max_tenor is None:
            max_tenor = self._get_max_tenor()
            
        tenors = np.linspace(0.05, max_tenor, 100)
        dfs = [self.get_discount_factor(t) for t in tenors]
        zero_rates = [self.get_zero_rate(t) * 100 for t in tenors]  # To percent
        
        # Calculate 3M forward rates
        fwd_tenors = tenors[:-4]  # Exclude last few points
        fwd_rates = []
        for t in fwd_tenors:
            try:
                fwd_rate = self.get_forward_rate(t, t + 0.25) * 100  # 3M forward
                fwd_rates.append(fwd_rate)
            except:
                fwd_rates.append(np.nan)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Discount factors
        ax1.plot(tenors, dfs, '-', color='blue', linewidth=2)
        ax1.set_xlabel('Tenor (years)')
        ax1.set_ylabel('Discount Factor')
        ax1.set_title(f'{title} - Discount Factors')
        ax1.grid(True)
        
        # Zero rates
        ax2.plot(tenors, zero_rates, '-', color='red', linewidth=2)
        ax2.set_xlabel('Tenor (years)')
        ax2.set_ylabel('Zero Rate (%)')
        ax2.set_title(f'{title} - Zero Rates')
        ax2.grid(True)
        
        # Forward rates
        ax3.plot(fwd_tenors, fwd_rates, '-', color='green', linewidth=2)
        ax3.set_xlabel('Start Tenor (years)')
        ax3.set_ylabel('3M Forward Rate (%)')
        ax3.set_title(f'{title} - 3M Forward Rates')
        ax3.grid(True)
        
        plt.tight_layout()
        return fig
    
    def _get_max_tenor(self):
        """Get maximum tenor for the curve - to be implemented by subclasses"""
        return 10.0


# In[6]:


get_ipython().system('jupyter nbconvert --to script abstract_curve.ipynb')

