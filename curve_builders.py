#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dataclasses import dataclass
from typing import List
from discount_curve import *
from zero_curve import *
from forward_curve import *
from utils import Interpolator, LogLinearInterpolator, MonotonicCubicInterpolator

# OISCurveBuilder class
# ForwardCurveBuilder class
# TreasuryCurveBuilder class


# In[2]:


class OISCurveBuilder:
    """Builds OIS discount curve from market instruments"""
    
    def __init__(self, instruments):
        """
        Initialize curve builder
        
        Args:
            instruments: List of OIS instruments (deposits and swaps)
        """
        self.instruments = instruments
        
    def build(self):
        """
        Build OIS discount curve
        
        Returns:
            DiscountCurve object
        """
        curve = DiscountCurve(interpolator=LogLinearInterpolator())
        
        # Initialize with DF(0,0) = 1.0
        curve.add_discount_factor(0.0, 1.0)
        
        # Sort instruments by maturity
        def get_maturity(instrument):
            return getattr(instrument, 'maturity', float('inf'))
            
        self.instruments.sort(key=get_maturity)
        
        print("\nBuilding OIS Discount Curve:")
        for i, instrument in enumerate(self.instruments):
            print(f"  {i+1}. Processing {instrument}")
            
            if hasattr(instrument, 'bootstrap'):
                instrument.bootstrap(curve)
            else:
                print(f"Warning: Skipping unsupported instrument type {type(instrument).__name__}")
                
        return curve


# In[3]:


class ForwardCurveBuilder:
    """Builds IBOR/SOFR forward curve from market instruments"""
    
    def __init__(self, instruments, discount_curve, index_name="SOFR", tenor_basis=0.25):
        """
        Initialize forward curve builder
        
        Args:
            instruments: List of forward curve instruments
            discount_curve: OIS discount curve for PV calculations
            index_name: Name of the index (e.g., "SOFR", "EURIBOR")
            tenor_basis: Tenor of the index in years (0.25 = 3M)
        """
        self.instruments = instruments
        self.discount_curve = discount_curve
        self.index_name = index_name
        self.tenor_basis = tenor_basis
        
    def build(self):
        """
        Build forward curve
        
        Returns:
            ForwardCurve object
        """
        curve = ForwardCurve(
            index_name=self.index_name,
            tenor_basis=self.tenor_basis,
            discount_curve=self.discount_curve,
            interpolator=MonotonicCubicInterpolator()
        )
        
        # Sort instruments by maturity
        def get_maturity(instrument):
            if hasattr(instrument, 'maturity'):
                return instrument.maturity
            elif hasattr(instrument, 'end_date'):
                return instrument.end_date
            return float('inf')
            
        self.instruments.sort(key=get_maturity)
        
        print(f"\nBuilding {self.index_name} Forward Curve:")
        for i, instrument in enumerate(self.instruments):
            print(f"  {i+1}. Processing {instrument}")
            
            if hasattr(instrument, 'bootstrap'):
                if isinstance(instrument, (IBORDeposit, SOFRFuture)):
                    instrument.bootstrap(curve)
                elif isinstance(instrument, IBORSwap):
                    instrument.bootstrap(curve, self.discount_curve)
                else:
                    print(f"Warning: Unsupported instrument type {type(instrument).__name__}")
            else:
                print(f"Warning: Skipping instrument without bootstrap method: {type(instrument).__name__}")
                
        return curve


# In[4]:


class TreasuryCurveBuilder:
    """Builds Treasury zero curve from market instruments"""
    
    def __init__(self, instruments):
        """
        Initialize Treasury curve builder
        
        Args:
            instruments: List of Treasury bonds and bills
        """
        self.instruments = instruments
        
    def build(self):
        """
        Build Treasury zero curve
        
        Returns:
            ZeroCurve object
        """
        curve = ZeroCurve(interpolator=MonotonicCubicInterpolator())
        
        # Sort instruments by maturity
        def get_maturity(instrument):
            return getattr(instrument, 'maturity', float('inf'))
            
        self.instruments.sort(key=get_maturity)
        
        print("\nBuilding Treasury Zero Curve:")
        for i, instrument in enumerate(self.instruments):
            print(f"  {i+1}. Processing {instrument}")
            
            if isinstance(instrument, TreasuryBond):
                instrument.bootstrap(curve)
            else:
                print(f"Warning: Skipping unsupported instrument: {type(instrument).__name__}")
                
        return curve


# In[6]:


get_ipython().system('jupyter nbconvert --to script curve_builders.ipynb')

