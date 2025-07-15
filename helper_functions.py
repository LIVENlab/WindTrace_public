import numpy as np
import scipy
from typing import Tuple, Dict
import pandas as pd
import matplotlib.pyplot as plt

def test_normal(residuals: np.array, material: str, print_acceptance: bool) -> bool:
    """Tests if residuals follow a normal distribution using Shapiro-Wilk test."""
    if residuals.size < 3:  # Shapiro-Wilk requires at least 3 data points
        print(f"Skipping Normality test for {material}: Not enough data points ({residuals.size}).")
        return False
    stat, p_value = scipy.stats.shapiro(residuals)
    if p_value > 0.05:
        if print_acceptance:
            print(f"Normality test for {material}: Accepted (p={p_value:.3f})")
        return True
    if print_acceptance:
        print(f"Normality test for {material}: Declined (p={p_value:.3f})")
    return False


def test_lognormal(residuals: np.array, material: str, print_acceptance: bool) -> bool:
    """Tests if residuals follow a lognormal distribution."""
    # Add small offset to ensure positive values for log transformation
    adjusted_residuals = residuals - np.min(residuals) + 1e-9
    # If all adjusted residuals are effectively zero or negative after adjustment, skip
    if np.any(adjusted_residuals <= 0) or adjusted_residuals.size < 3:
        print(f"Skipping Lognormality test for {material}: Invalid or insufficient data for log transformation.")
        return False

    log_residuals = np.log(adjusted_residuals)
    stat, p_value = scipy.stats.shapiro(log_residuals)
    if p_value > 0.05:
        if print_acceptance:
            print(f"Lognormality test for {material}: Accepted (p={p_value:.3f})")
        return True
    if print_acceptance:
        print(f"Lognormality test for {material}: Declined (p={p_value:.3f})")
    return False


def test_triangular(residuals: np.array, material: str, print_acceptance: bool) -> bool:
    """Tests if residuals follow a triangular distribution using Kolmogorov-Smirnov test."""
    if residuals.size < 2:
        print(f"Skipping Triangular test for {material}: Not enough data points ({residuals.size}).")
        return False
    min_res = np.min(residuals)
    max_res = np.max(residuals)
    if min_res == max_res:  # Cannot fit a triangular distribution with zero range
        print(f"Skipping Triangular test for {material}: Residual range is zero.")
        return False
    c = (min_res + max_res) / 2  # Mode at midpoint
    loc = min_res
    scale = max_res - min_res
    _, p_value = scipy.stats.kstest(residuals, 'triang', args=((c - loc) / scale, loc, scale))
    if p_value > 0.05:
        if print_acceptance:
            print(f"Triangular test for {material}: Accepted (p={p_value:.3f})")
        return True
    if print_acceptance:
        print(f"Triangular test for {material}: Declined (p={p_value:.3f})")
    return False


def test_uniform(residuals: np.array, material: str, print_acceptance: bool) -> bool:
    """Tests if residuals follow a uniform distribution using Kolmogorov-Smirnov test."""
    if residuals.size < 2:
        print(f"Skipping Uniform test for {material}: Not enough data points ({residuals.size}).")
        return False
    min_res = np.min(residuals)
    max_res = np.max(residuals)
    if min_res == max_res:  # Cannot fit a uniform distribution with zero range
        print(f"Skipping Uniform test for {material}: Residual range is zero.")
        return False
    _, p_value = scipy.stats.kstest(residuals, 'uniform', args=(min_res, max_res - min_res))
    if p_value > 0.05:
        if print_acceptance:
            print(f"Uniform test for {material}: Accepted (p={p_value:.3f})")
        return True
    if print_acceptance:
        print(f"Uniform test for {material}: Declined (p={p_value:.3f})")
    return False


def test_residual_distributions(residuals: np.array, material: str, print_acceptance: bool = False) -> str:
    """Tests residuals against common distributions and returns the best fit."""
    # Ensure residuals are not empty
    if residuals.size == 0:
        print(f"No residuals to test for {material}.")
        return 'none'

    # Test distributions in order of preference
    if test_normal(residuals, material, print_acceptance):
        return 'normal'
    if test_lognormal(residuals, material, print_acceptance):
        return 'lognormal'
    if test_triangular(residuals, material, print_acceptance):
        return 'triangular'
    if test_uniform(residuals, material, print_acceptance):
        return 'uniform'
    return 'none'


def statistical_results(residuals: np.array) -> Tuple[float, float, float]:
    """Calculates confidence interval, standard deviation, and variance from residuals."""
    if residuals.size == 0:
        return np.nan, np.nan, np.nan
    std_error = np.sqrt(np.mean(residuals ** 2))
    confidence = 1.96 * std_error  # 95% confidence interval multiplier (for large n)
    residual_variance = np.mean(residuals ** 2)
    residual_std_dev = np.sqrt(residual_variance)
    return confidence, residual_std_dev, residual_variance


def fit_model(x: np.array, y: np.array, material_name: str,
              proportional: bool = False) -> Dict:
    """
    Fits a linear regression model, calculates statistics, and tests residual distributions.
    Returns a dictionary with 'polyfit', 'confidence_95%', 'std_dev', and 'residual_distribution'.
    Includes robustness checks for insufficient or degenerate data.
    """
    if len(x) < 2 or len(y) < 2:
        print(
            f"Warning: Not enough valid data points ({len(x)}) to fit model for {material_name}. Returning dummy model.")
        return {
            'polyfit': np.poly1d([0]),
            'confidence_95%': np.nan,
            'std_dev': np.nan,
            'residual_distribution': 'none'
        }

    poly = None
    if proportional:
        if np.sum(x ** 2) == 0:  # Avoid division by zero
            slope = 0
        else:
            slope = np.sum(x * y) / np.sum(x ** 2)
        poly = np.poly1d([slope, 0])  # y = mx + 0
    else:
        try:
            poly = np.poly1d(np.polyfit(x, y, 1))  # y = mx + b
        except np.linalg.LinAlgError:
            print(
                f"Warning: Linear algebra error during polyfit for {material_name}. Data might be degenerate (e.g., all x values identical). Returning mean as constant model.")
            poly = np.poly1d([np.mean(y)])  # Fallback to a constant model (mean of y)
        except Exception as e:
            print(
                f"Warning: An unexpected error occurred during polyfit for {material_name}: {e}. Returning dummy model.")
            poly = np.poly1d([0])  # Fallback to a constant zero model

    if poly is None:  # Safeguard
        poly = np.poly1d([0])

    # Calculate residuals. Ensure poly is callable and x is not empty.
    residuals = np.array([])
    if len(x) > 0 and isinstance(poly, np.poly1d):
        try:
            residuals = y - poly(x)
        except Exception as e:
            print(f"Error calculating residuals for {material_name}: {e}. Residuals will be empty.")
            residuals = np.array([])

    # Perform statistical tests and calculations only if residuals are available
    if residuals.size == 0:
        dist_type = 'none'
        confidence = np.nan
        std_dev = np.nan
        residual_variance = np.nan
    else:
        dist_type = test_residual_distributions(residuals, material_name, False)
        if dist_type == 'lognormal':
            # Apply log transformation for statistical results if distribution is lognormal
            adjusted_residuals = residuals - np.min(residuals) + 1e-9
            if adjusted_residuals.size > 0 and np.all(adjusted_residuals > 0):
                confidence, std_dev, residual_variance = statistical_results(np.log(adjusted_residuals))
            else:
                # Fallback if log transformation is problematic
                print(
                    f"Warning: Log transformation for statistical results failed for {material_name}. Using original residuals.")
                confidence, std_dev, residual_variance = statistical_results(residuals)
        else:
            confidence, std_dev, residual_variance = statistical_results(residuals)

    return {
        'polyfit': poly,
        'confidence_95%': confidence,
        'std_dev': std_dev,
        'residual_distribution': dist_type
    }


def plot_qq_plot(residuals: np.array, plot_title: str, dist_type: str):
    """
    Generates a Q-Q plot for the given residuals against the specified distribution type.
    """
    if residuals.size < 3:  # scipy.stats.probplot needs at least 3 points
        print(f"Skipping Q-Q plot for {plot_title}: Not enough residuals ({residuals.size}) to plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    if dist_type == 'lognormal':
        # Apply log transformation to residuals for lognormal Q-Q plot
        adjusted_residuals = residuals - np.min(residuals) + 1e-9
        if adjusted_residuals.size > 0 and np.all(adjusted_residuals > 0):
            scipy.stats.probplot(np.log(adjusted_residuals), dist="norm", plot=ax)
            ax.set_title(f'Q-Q Plot: {plot_title} (Lognormal reference)')
        else:
            print(
                f"Skipping Q-Q plot for {plot_title}: No positive residuals for lognormal reference after adjustment.")
            plt.close(fig)
            return
    else:
        # Default to normal distribution for other cases or when no specific fit
        scipy.stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot: {plot_title} (Normal reference)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def calculate_intersection(poly1: np.poly1d, poly2: np.poly1d) -> np.array:
    """Calculates real intersection points between two polynomials."""
    intersection_poly = np.poly1d(poly1 - poly2)
    intersection_x = np.roots(intersection_poly)
    real_intersection_x = intersection_x[np.isreal(intersection_x)].real
    return real_intersection_x if real_intersection_x.size > 0 else np.array([])


def load_vestas_data(file_path: str, sheet_name: str) -> pd.DataFrame:
    """Loads Vestas turbine materials data from a specified Excel sheet."""
    return pd.read_excel(file_path, sheet_name=sheet_name, dtype=None, decimal=";", header=0)


def plot_materials(x, y, residuals, interpolation_eq, confidence, xlabel: str, ylabel: str, title: str, grid=True,
                   adjusted_plot=True):
    """
    For the scatter points of x and y, given the residuals, fitting curve (interpolation_eq), and confidence 95% (value
    that stablishes the minimim and maximum deviation from the mean that guarantees that 95% of the values will fall in
    that range), it shows the corresponding plot. It's not saving it, just showing.
    Note:
    The variable adjusted_plot allows to extend the plot from 0 to 15 MW.
    """
    y_mean = np.mean(y)
    # Handle case where ss_total might be zero (e.g., all y values are the same)
    ss_total = np.sum((y - y_mean) ** 2)
    ss_residual = np.sum(residuals ** 2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else np.nan

    # Determine x interpolation range
    if adjusted_plot:
        # Check if x has at least two unique points to define a range
        if len(np.unique(x)) < 2:
            x_interpolate = np.array([x.min(), x.max()]) if x.size > 0 else np.array([0, 1])
        else:
            x_interpolate = np.linspace(x.min(), x.max(), 100)
    else:
        x_interpolate = np.linspace(0, 15, 100)

    # Ensure interpolation_eq is a callable poly1d object before calling
    if isinstance(interpolation_eq, np.poly1d):
        y_interpolated = interpolation_eq(x_interpolate)
    else:
        y_interpolated = np.zeros_like(x_interpolate)  # Fallback if polyfit failed

    # Plot the data, fitted curve, and confidence interval
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Data')
    plt.plot(x_interpolate, y_interpolated, color='red', label='Regression Line')

    # Only plot confidence interval if confidence is a valid number
    if not np.isnan(confidence):
        plt.fill_between(x_interpolate, y_interpolated - confidence, y_interpolated + confidence,
                         color='red', alpha=0.2, label='95% Confidence Interval')

    # Add R-squared annotation
    if not np.isnan(r_squared):
        plt.annotate(f"R-squared = {r_squared:.2f}", xy=(0.05, 0.9), xycoords='axes fraction', fontsize=10)
    else:
        plt.annotate("R-squared: N/A", xy=(0.05, 0.9), xycoords='axes fraction', fontsize=10)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(grid)
    plt.legend()
    plt.show()