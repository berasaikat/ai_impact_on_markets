import pandas as pd
import numpy as np
import logging
from scipy.interpolate import griddata
from config import REALIZED_VOL_WINDOW, VIX_LOW_THRESHOLD, VIX_HIGH_THRESHOLD

def realized_vol(returns_df: pd.DataFrame, window: int = REALIZED_VOL_WINDOW, annualize: bool = True) -> pd.DataFrame:
    """
    Compute rolling realised volatility for each column of a returns DataFrame.
    """
    vol = returns_df.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(252)
        
    vol = vol.dropna()
    vol.columns = [f"{str(col)}_RVol" for col in vol.columns]
    
    return vol

def vol_spread(realized: pd.Series, iv_atm: pd.Series) -> pd.Series:
    """
    Compute the volatility risk premium spread: IV_ATM - Realized_Vol.
    """
    if realized.empty or iv_atm.empty:
        return pd.Series(dtype=float, name='VolSpread')
        
    if realized.mean() > 5.0 or iv_atm.mean() > 5.0:
        raise ValueError("Series mean > 5.0. Both series must be in decimal annualised units.")
        
    spread = iv_atm - realized
    spread.name = 'VolSpread'
    
    return spread

def build_vol_surface(iv_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate the raw (irregular) IV matrix onto a regular grid suitable for 3D plotting.
    """
    if iv_matrix.shape[0] < 3 or iv_matrix.shape[1] < 2:
        logging.warning("iv_matrix has fewer than 3 strikes or fewer than 2 expiries. Returning unchanged.")
        return iv_matrix
        
    today = pd.Timestamp('today').normalize()
    expiries = pd.to_datetime(iv_matrix.columns)
    dte_series = (expiries - today).days
    
    points = []
    values = []
    
    for col_idx, col_name in enumerate(iv_matrix.columns):
        dte = dte_series[col_idx]
        for strike in iv_matrix.index:
            val = iv_matrix.loc[strike, col_name]
            if pd.notna(val):
                points.append((strike, dte))
                values.append(val)
                
    if not points:
        return iv_matrix
        
    points = np.array(points)
    values = np.array(values)
    
    target_strikes = np.linspace(iv_matrix.index.min(), iv_matrix.index.max(), 50)
    
    num_expiries = min(len(iv_matrix.columns), 8)
    target_dte = np.linspace(dte_series.min(), dte_series.max(), num_expiries)
    target_dte = np.round(target_dte).astype(int)
    
    grid_strikes, grid_dte = np.meshgrid(target_strikes, target_dte, indexing='ij')
    
    grid_z = griddata(points, values, (grid_strikes, grid_dte), method='cubic')
    
    grid_z_nearest = griddata(points, values, (grid_strikes, grid_dte), method='nearest')
    grid_z = np.where(np.isnan(grid_z), grid_z_nearest, grid_z)
    
    res_df = pd.DataFrame(grid_z, index=target_strikes, columns=target_dte)
    return res_df

def vol_regime(vix_series: pd.Series, low_threshold: float = VIX_LOW_THRESHOLD, high_threshold: float = VIX_HIGH_THRESHOLD) -> pd.Series:
    """
    Label each day as 'low', 'normal', or 'high' volatility regime based on VIX level.
    """
    if vix_series.empty:
        return pd.Series(dtype='category', name='vol_regime')
        
    conditions = [
        vix_series < low_threshold,
        vix_series > high_threshold
    ]
    choices = ['low', 'high']
    regime_arr = np.select(conditions, choices, default='normal')
    
    regime = pd.Series(regime_arr, index=vix_series.index, name='vol_regime')
    cat_type = pd.CategoricalDtype(categories=['low', 'normal', 'high'], ordered=True)
    regime = regime.astype(cat_type)
    
    return regime
