# -*- coding: utf-8 -*-
"""
Visualization functions for the trajectory segmentation pipeline.

Includes plotting segmentation results, cost curves, orientation deviation,
and GMM waypoint optimization details.

REVISED import logic for orientation functions.
ADDED XYZ component plot and 12D pose component plot.
FIXED XYZ plot Y-axis scaling.
RENAMED 3D trajectory comparison plot to use waypoints.
ADDED plot for optimization steps.
FIXED dimension mismatch error in plot_optimization_step_details_3d scaling.
UPDATED optimization step path style for better visibility.
"""

# --- Standard Library Imports ---
import math
import sys
import traceback # For detailed error printing
import random # For unique labels if needed
from pathlib import Path
from itertools import combinations
from typing import Optional, Dict, Any, List # Added for type hinting clarity

# --- Third-Party Library Imports ---
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D # For legend proxies
from mpl_toolkits.mplot3d import Axes3D # For 3D plots
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # For task space set
import numpy as np
import seaborn as sns
import yaml # For loading cross-section results if needed
import plotly.graph_objects as go

# --- Project Modules & Dependencies ---
# Attempt to import project-specific modules and check for dependencies
try:
    import config # Import configuration
    import utils # Import utils for orientation math, FK
    from utils import IK_SOLVER_AVAILABLE, SCIPY_AVAILABLE

    # Import linalg if available for ellipsoid plotting
    if SCIPY_AVAILABLE:
        # Use a more specific import to avoid potential namespace conflicts
        from scipy import linalg as scipy_linalg
    else:
        scipy_linalg = None # Define as None if not available
    print("INFO (visualizer.py): Successfully imported config and utils.")

except ImportError:
    print("FATAL ERROR (visualizer.py): config.py or utils.py not found.")
    print("                     Proceeding with dummy implementations and disabled features.")
    # Define dummy flags and a dummy utils class if imports fail
    IK_SOLVER_AVAILABLE = False
    SCIPY_AVAILABLE = False
    scipy_linalg = None

    class DummyUtils:
        """Dummy class to provide placeholders if utils fails to import."""
        SCIPY_AVAILABLE = False
        IK_SOLVER_AVAILABLE = False
        def forward_kinematics(self, *args, **kwargs): return None, None, None
        def log_map_so3(self, *args, **kwargs): return None
        def calculate_geodesic_distance(self, *args, **kwargs): return np.nan
        def exp_map_so3(self, *args, **kwargs): return None
        def calculate_tcp_trajectory(self, *args, **kwargs): return None # Add dummy

    utils = DummyUtils() # Instantiate the dummy class

# --- Define local placeholders for orientation functions ---
_log_map_so3_func = None
_exp_map_so3_func = None

if SCIPY_AVAILABLE and hasattr(utils, 'log_map_so3') and hasattr(utils, 'exp_map_so3'):
    try:
        # Directly assign functions from the imported utils module
        log_map_so3_util = getattr(utils, 'log_map_so3')
        exp_map_so3_util = getattr(utils, 'exp_map_so3')

        if callable(log_map_so3_util):
            _log_map_so3_func = log_map_so3_util
        if callable(exp_map_so3_util):
            _exp_map_so3_func = exp_map_so3_util

        if _log_map_so3_func and _exp_map_so3_func:
            print("INFO (visualizer.py): Successfully linked orientation helpers from utils.")
        else:
            print("ERROR (visualizer.py): One or more orientation helpers from utils are not callable!")
            _log_map_so3_func = None
            _exp_map_so3_func = None
    except Exception as e:
        print(f"ERROR (visualizer.py): Failed to link orientation helpers due to: {e}")
        _log_map_so3_func = None
        _exp_map_so3_func = None
elif not SCIPY_AVAILABLE:
    print("INFO (visualizer.py): Scipy not available via utils. Orientation functions disabled.")
else:
    print("INFO (visualizer.py): Orientation functions (log_map_so3, exp_map_so3) not found in utils module.")


# --- Plotting Helper Functions ---

def _get_ellipse_plotly(mean_2d: np.ndarray, cov_2d: np.ndarray, n_std: float = 2.0, n_points: int = 50) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generates x, y points for a 2D confidence ellipse for Plotly visualization.

    Args:
        mean_2d: The 2D mean vector (shape [2,]).
        cov_2d: The 2x2 covariance matrix.
        n_std: The number of standard deviations for the ellipse size.
        n_points: The number of points to generate for the ellipse boundary.

    Returns:
        A tuple (x_points, y_points) for the ellipse, or (None, None) on error.
    """
    try:
        # Ensure covariance matrix is symmetric
        cov_2d = (cov_2d + cov_2d.T) / 2
        vals, vecs = np.linalg.eigh(cov_2d)

        # Check for non-positive eigenvalues (indicates degenerate ellipse)
        if np.any(vals <= 1e-9):
            print("Warning (_get_ellipse_plotly): Non-positive eigenvalues found in covariance.")
            return None, None

        # Calculate angle and dimensions
        angle = np.arctan2(vecs[1, 0], vecs[0, 0]) # More robust angle calculation
        width, height = 2 * n_std * np.sqrt(vals) # Eigenvalues are variances

        # Generate points on the standard ellipse
        t = np.linspace(0, 2 * np.pi, n_points)
        xs_unit = width / 2 * np.cos(t)
        ys_unit = height / 2 * np.sin(t)

        # Rotation matrix
        R_mat = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

        # Rotate and translate points
        points_rotated = R_mat @ np.vstack([xs_unit, ys_unit])
        x_points = points_rotated[0, :] + mean_2d[0]
        y_points = points_rotated[1, :] + mean_2d[1]

        return x_points, y_points

    except np.linalg.LinAlgError as e:
        print(f"Error (_get_ellipse_plotly): Linear algebra error: {e}")
        return None, None
    except Exception as e:
        print(f"Error (_get_ellipse_plotly): Unexpected error: {e}")
        return None, None

# --- Main Plotting Functions ---

def plot_gmm_cross_section_interactive(
    gmm_params: Optional[dict],
    snapshot_data: Optional[np.ndarray],
    feature_names: Optional[List[str]] = None,
    title: str = "Interactive GMM Cross-Section Visualization"
):
    """
    Creates an interactive Plotly plot showing pairwise projections of GMM components.

    Args:
        gmm_params: Dictionary containing GMM parameters ('weights', 'means', 'covariances', 'n_components_used').
        snapshot_data: Numpy array of snapshot data points (shape [n_samples, n_features]).
        feature_names: List of names for each feature dimension. Defaults if None.
        title: The title for the Plotly figure.
    """
    # Default feature names if not provided
    if feature_names is None:
        feature_names = ['px', 'py', 'pz', 'vlog_x', 'vlog_y', 'vlog_z']

    print(f"--- Generating Interactive GMM Plot: {title} ---")

    # --- Input Validation ---
    if gmm_params is None:
        print("  Error: gmm_params dictionary is None. Cannot generate plot.")
        return
    if snapshot_data is None:
        print("  Error: snapshot_data is None. Cannot generate plot.")
        return
    if not isinstance(snapshot_data, np.ndarray) or snapshot_data.ndim != 2 or snapshot_data.shape[0] == 0:
        print(f"  Error: snapshot_data is invalid (shape: {snapshot_data.shape if isinstance(snapshot_data, np.ndarray) else 'N/A'}, ndim: {snapshot_data.ndim if isinstance(snapshot_data, np.ndarray) else 'N/A'}). Expected a 2D numpy array with data.")
        return

    try:
        # --- Extract GMM Parameters ---
        weights = np.array(gmm_params.get('weights', []))
        means = np.array(gmm_params.get('means', []))
        covariances = np.array(gmm_params.get('covariances', []))
        n_components = gmm_params.get('n_components_used', len(weights))
        n_features = len(feature_names)

        # --- Dimension Checks ---
        if weights.shape != (n_components,):
            print(f"  Error: GMM weights dimension mismatch. Expected ({n_components},), got {weights.shape}.")
            return
        if means.shape != (n_components, n_features):
            print(f"  Error: GMM means dimension mismatch. Expected ({n_components}, {n_features}), got {means.shape}.")
            return
        if covariances.shape != (n_components, n_features, n_features):
            print(f"  Error: GMM covariances dimension mismatch. Expected ({n_components}, {n_features}, {n_features}), got {covariances.shape}.")
            return
        if snapshot_data.shape[1] != n_features:
            print(f"  Error: snapshot_data feature dimension ({snapshot_data.shape[1]}) does not match expected ({n_features}).")
            return

        # --- Plotly Figure Initialization ---
        fig = go.Figure()
        dimension_indices = list(range(n_features))
        pairwise_combinations = list(combinations(dimension_indices, 2))
        component_colors = sns.color_palette("viridis", n_components).as_hex()

        print(f"  Generating traces for {len(pairwise_combinations)} dimension pairs and {n_components} components...")

        # --- Generate Traces for Each Pairwise Combination ---
        for i, (dim_x_idx, dim_y_idx) in enumerate(pairwise_combinations):
            # Set visibility: only the first pair is visible initially
            visible = (i == 0)

            # Add snapshot data trace (only needs legend entry once)
            fig.add_trace(go.Scatter(
                x=snapshot_data[:, dim_x_idx],
                y=snapshot_data[:, dim_y_idx],
                mode='markers',
                marker=dict(color='grey', size=5, opacity=0.7),
                name='Snapshots',
                visible=visible,
                showlegend=(i == 0) # Show legend only for the first set of traces
            ))

            # Add ellipse traces for each GMM component
            for k in range(n_components):
                mean_2d = means[k, [dim_x_idx, dim_y_idx]]
                cov_2d = covariances[k, np.ix_([dim_x_idx, dim_y_idx], [dim_x_idx, dim_y_idx])]

                x_ellipse, y_ellipse = _get_ellipse_plotly(mean_2d, cov_2d, n_std=2.0)

                if x_ellipse is not None and y_ellipse is not None:
                    fig.add_trace(go.Scatter(
                        x=x_ellipse,
                        y=y_ellipse,
                        mode='lines',
                        line=dict(color=component_colors[k], width=2),
                        fill='toself',
                        fillcolor=component_colors[k],
                        opacity=0.3 + 0.6 * weights[k], # Opacity scaled by weight
                        name=f'Comp {k+1} (w={weights[k]:.2f})',
                        visible=visible,
                        showlegend=(i == 0) # Show legend only for the first set
                    ))
                else:
                    # Add a dummy trace if ellipse generation failed, to keep indices consistent
                    fig.add_trace(go.Scatter(
                        x=[None], y=[None], mode='markers',
                        name=f'Comp {k+1} Error', visible=False, showlegend=False
                    ))

        # --- Create Dropdown Buttons for Dimension Selection ---
        buttons = []
        traces_per_pair = 1 + n_components # 1 snapshot trace + n_components ellipse traces
        for i, (dim_x_idx, dim_y_idx) in enumerate(pairwise_combinations):
            # Create a visibility mask for this dimension pair
            visibility_mask = [False] * len(fig.data)
            start_idx = i * traces_per_pair
            end_idx = start_idx + traces_per_pair

            # Mark traces belonging to this pair as visible (if they have valid data)
            for trace_idx in range(start_idx, end_idx):
                if trace_idx < len(visibility_mask):
                    # Check if the trace has valid data before making it visible
                    trace_data = fig.data[trace_idx]
                    has_data = (trace_data.x is not None and
                                len(trace_data.x) > 0 and
                                trace_data.x[0] is not None)
                    if has_data:
                        visibility_mask[trace_idx] = True

            buttons.append(dict(
                label=f"{feature_names[dim_x_idx]} vs {feature_names[dim_y_idx]}",
                method="update",
                args=[
                    {"visible": visibility_mask},
                    {"xaxis.title": feature_names[dim_x_idx],
                     "yaxis.title": feature_names[dim_y_idx]}
                ]
            ))

        # --- Update Layout with Dropdown and Titles ---
        fig.update_layout(
            updatemenus=[dict(
                active=0,
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.05, # Position dropdown slightly to the right
                xanchor="left",
                y=1.15, # Position dropdown above the plot
                yanchor="top"
            )],
            title=title,
            hovermode="closest" # Better hover interaction
        )

        # Set initial axis labels based on the first visible pair
        initial_dim_x_idx, initial_dim_y_idx = pairwise_combinations[0]
        fig.update_layout(
            xaxis_title=feature_names[initial_dim_x_idx],
            yaxis_title=feature_names[initial_dim_y_idx],
            legend_title_text="Components"
        )

        print("  Displaying interactive plot...")
        fig.show() # Display the plot

    except Exception as e:
        print(f"  Error generating interactive GMM plot: {e}")
        traceback.print_exc()


def plot_cost_vs_segments(
    raw_costs: np.ndarray,
    max_segments: int,
    optimal_num_segments: int,
    lambda_penalty: float
):
    """
    Plots the raw and penalized segmentation cost versus the number of segments.

    Args:
        raw_costs: Array of raw costs for 1 to N segments.
        max_segments: The maximum number of segments considered.
        optimal_num_segments: The determined optimal number of segments.
        lambda_penalty: The penalty factor used.
    """
    print("--- Generating Cost vs. Segments Plot ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    # Use config font if available, otherwise default
    plot_font = getattr(config, 'PLOT_FONT', 'sans-serif')
    plt.rcParams['font.family'] = plot_font

    num_segments_axis = np.arange(1, max_segments + 1)

    # Initialize cost arrays with infinity, then fill with valid costs
    plot_costs_raw = np.full(max_segments, np.inf)
    valid_len = min(len(raw_costs), max_segments)
    if valid_len > 0:
        plot_costs_raw[:valid_len] = raw_costs[:valid_len]

    # Calculate penalized costs (handle potential inf values in raw_costs)
    penalized_costs = np.where(np.isfinite(plot_costs_raw),
                               plot_costs_raw + lambda_penalty * num_segments_axis,
                               np.inf)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    color_raw = 'tab:blue'
    ax1.set_xlabel('Number of Segments')
    ax1.set_ylabel('Total Raw Segmentation Cost', color=color_raw)

    # Plot raw costs if finite values exist
    valid_idx_raw = np.isfinite(plot_costs_raw)
    if np.any(valid_idx_raw):
        ax1.plot(num_segments_axis[valid_idx_raw], plot_costs_raw[valid_idx_raw],
                 marker='o', linestyle='-', color=color_raw, label='Raw Cost')
        ax1.tick_params(axis='y', labelcolor=color_raw)
        min_finite_cost_raw = np.min(plot_costs_raw[valid_idx_raw])
        # Adjust y-axis lower limit for better visibility
        ax1.set_ylim(bottom=min(0, min_finite_cost_raw * 0.9 if min_finite_cost_raw >= 0 else min_finite_cost_raw * 1.1))
    else:
        # Display message if no valid raw costs
        ax1.text(0.5, 0.5, 'No finite raw costs to plot',
                 ha='center', va='center', transform=ax1.transAxes)

    # Create a second y-axis for penalized costs
    ax2 = ax1.twinx()
    color_penalized = 'tab:red'
    ax2.set_ylabel('Total Penalized Cost (Raw + λ*N)', color=color_penalized)

    # Plot penalized costs if finite values exist
    valid_idx_pen = np.isfinite(penalized_costs)
    if np.any(valid_idx_pen):
        ax2.plot(num_segments_axis[valid_idx_pen], penalized_costs[valid_idx_pen],
                 marker='x', linestyle='--', color=color_penalized, label='Penalized Cost')
        ax2.tick_params(axis='y', labelcolor=color_penalized)
        # Calculate appropriate y-limits for penalized cost
        min_finite_cost_pen = np.min(penalized_costs[valid_idx_pen])
        max_finite_cost_pen = np.max(penalized_costs[valid_idx_pen])
        padding = (max_finite_cost_pen - min_finite_cost_pen) * 0.05 if max_finite_cost_pen > min_finite_cost_pen else 1.0
        ax2.set_ylim(bottom=min(0, min_finite_cost_pen) - padding,
                     top=max_finite_cost_pen + padding)
    else:
        # Display message if no valid penalized costs
        ax2.text(0.5, 0.4, 'No finite penalized costs to plot',
                 ha='center', va='center', transform=ax2.transAxes)

    # Highlight the optimal number of segments
    if 1 <= optimal_num_segments <= max_segments:
        optimal_raw_cost_idx = optimal_num_segments - 1
        optimal_raw_cost = plot_costs_raw[optimal_raw_cost_idx]
        optimal_pen_cost = penalized_costs[optimal_raw_cost_idx]

        # Scatter plot for optimal raw cost (if finite)
        if np.isfinite(optimal_raw_cost):
            ax1.scatter(optimal_num_segments, optimal_raw_cost, color='green', s=150,
                        zorder=5, marker='*', label=f'Optimal ({optimal_num_segments}) - Raw')
        # Scatter plot for optimal penalized cost (if finite)
        if np.isfinite(optimal_pen_cost):
            ax2.scatter(optimal_num_segments, optimal_pen_cost, color='magenta', s=150,
                        zorder=5, marker='P', label=f'Optimal ({optimal_num_segments}) - Penalized')

    # --- Final Plot Adjustments ---
    lambda_val_str = f"{config.LAMBDA_PENALTY:.2e}" if hasattr(config, "LAMBDA_PENALTY") else "N/A"
    plt.title(f'Segmentation Cost vs. Number of Segments (λ={lambda_val_str})')
    ax1.set_xticks(np.arange(1, max_segments + 1)) # Ensure integer ticks for segment numbers
    ax1.grid(True, linestyle=':', which='major', axis='x') # Grid only on primary x-axis
    ax2.grid(False) # Turn off grid for the secondary y-axis

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.99, 0.95))

    # Adjust layout to prevent legend overlap
    fig.tight_layout(rect=[0, 0, 0.9, 1]) # Leave space on the right for the legend

    # Note: plt.show() should be called externally if needed immediately.


def plot_interface_frame_pose_components(
    tcp_pos_path_interface_np: Optional[np.ndarray],
    tcp_rot_matrices_interface_list: Optional[List[np.ndarray]], # List of 3x3 rotation matrices
    segment_boundary_indices: List[int],
    segment_stats_list: List[dict], # List of dicts (can be empty, currently unused in plot)
    segment_start_end_pose_components: List[dict], # List of dicts (currently unused in plot)
    title: str = "TCP Pose Components vs Waypoint Index (Interface Frame - LogMap)",
    log_map_feature_names: List[str] = ['vlog_x', 'vlog_y', 'vlog_z'] # Names for log map components
):
    """
    Plots the 3 position components (tx, ty, tz) and optionally the 3 orientation
    log map vector components of the TCP trajectory in the Interface frame
    against the waypoint index.

    Args:
        tcp_pos_path_interface_np: Numpy array of TCP positions (N, 3).
        tcp_rot_matrices_interface_list: List of N rotation matrices (3x3 numpy arrays).
        segment_boundary_indices: List of indices where segments end.
        segment_stats_list: Currently unused in this plot.
        segment_start_end_pose_components: Currently unused in this plot.
        title: Title for the plot.
        log_map_feature_names: Names for the log map components.
    """
    print("--- Generating Interface Frame Pose Component Plot (Position + Log Map) ---")

    # --- Input Validation and Setup ---
    if tcp_pos_path_interface_np is None or len(tcp_pos_path_interface_np) == 0:
        print("  Warning: No TCP position data provided. Cannot generate plot.")
        return

    num_waypoints = len(tcp_pos_path_interface_np)
    time_axis = np.arange(num_waypoints)

    # Check if rotation data is consistent with position data
    if tcp_rot_matrices_interface_list is None or len(tcp_rot_matrices_interface_list) != num_waypoints:
        print(f"  Warning: Rotation matrix list is None or length mismatch ({len(tcp_rot_matrices_interface_list) if tcp_rot_matrices_interface_list else 'None'} vs {num_waypoints}). Orientation components will not be plotted.")
        tcp_rot_matrices_interface_list = None # Disable rotation plotting

    # Check if the log map function is available
    log_map_available = callable(_log_map_so3_func)
    if not log_map_available:
        print("  Warning: log_map_so3 function unavailable. Cannot plot orientation components.")
        tcp_rot_matrices_interface_list = None # Disable rotation plotting

    # --- Calculate Log Map Trajectory (if possible) ---
    log_map_trajectory = None
    log_map_valid = False
    if tcp_rot_matrices_interface_list is not None and log_map_available:
        log_map_list = []
        calculation_errors = 0
        invalid_matrix_errors = 0
        valid_rots_found = 0

        for idx, R_mat in enumerate(tcp_rot_matrices_interface_list):
            v_log = None
            is_valid_rot = False

            # Validate the rotation matrix
            if R_mat is not None and isinstance(R_mat, np.ndarray) and R_mat.shape == (3,3) and not np.isnan(R_mat).any() and not np.isinf(R_mat).any():
                try:
                    det = np.linalg.det(R_mat)
                    is_ortho = np.allclose(R_mat @ R_mat.T, np.identity(3), atol=1e-4)
                    if abs(det - 1.0) < 1e-4 and is_ortho:
                        is_valid_rot = True
                    else:
                        # print(f"  Debug: Matrix {idx} invalid (det={det:.3f}, ortho={is_ortho})") # Optional debug
                        invalid_matrix_errors += 1
                except np.linalg.LinAlgError:
                    # print(f"  Debug: Matrix {idx} LinAlgError") # Optional debug
                    invalid_matrix_errors += 1
            else:
                # print(f"  Debug: Matrix {idx} invalid type/shape/content") # Optional debug
                invalid_matrix_errors += 1

            # Calculate log map if the matrix is valid
            if is_valid_rot:
                try:
                    v_log = _log_map_so3_func(R_mat) # Use the function handle
                    if v_log is None:
                        calculation_errors += 1
                        # print(f"  Debug: Log map calculation failed for index {idx}") # Optional debug
                    else:
                        valid_rots_found += 1
                except Exception as e:
                    calculation_errors += 1
                    # print(f"  Debug: Log map exception for index {idx}: {e}") # Optional debug

            # Append result (or NaNs if failed)
            log_map_list.append(v_log if v_log is not None else np.full(3, np.nan))

        if calculation_errors > 0: print(f"  Warning: {calculation_errors} log map calculations failed.")
        if invalid_matrix_errors > 0: print(f"  Warning: {invalid_matrix_errors} invalid rotation matrices encountered.")

        if valid_rots_found > 0:
            log_map_trajectory = np.array(log_map_list)
            log_map_valid = True
            print(f"  Successfully calculated log map for {valid_rots_found} waypoints.")
        else:
            print("  Warning: No valid log map vectors could be calculated.")

    # --- Determine Plot Layout ---
    n_features_pos = 3
    n_features_rot = 3 if log_map_valid else 0
    n_features_to_plot = n_features_pos + n_features_rot

    if n_features_to_plot == 0:
        print("  Error: No valid data (position or orientation) to plot.")
        return
    elif n_features_to_plot <= 3: # Only position or only orientation (less likely)
        ncols = n_features_to_plot
        nrows = 1
    else: # Position and orientation
        ncols = 3
        nrows = 2

    # --- Create Subplots ---
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5), sharex=True, squeeze=False)
    fig.suptitle(title, fontsize=14, y=0.99) # Adjust title position
    axs_flat = axs.flatten() # Flatten for easy iteration

    # --- Prepare Data and Feature Names ---
    position_feature_names = ['tx', 'ty', 'tz']
    all_feature_names = position_feature_names + (log_map_feature_names if log_map_valid else [])

    # Combine position and orientation data if available
    if log_map_valid:
        all_data = np.hstack((tcp_pos_path_interface_np, log_map_trajectory))
    else:
        all_data = tcp_pos_path_interface_np

    # --- Plot Each Component ---
    plotted_legend_labels = set() # Track labels to avoid duplicates in legend

    for d_idx, feature_name in enumerate(all_feature_names):
        if d_idx >= len(axs_flat): break # Stop if we run out of axes

        ax = axs_flat[d_idx]
        component_data = all_data[:, d_idx]
        valid_data_mask = ~np.isnan(component_data) # Mask for finite values

        # Plot TCP component path
        label_comp = 'TCP Path Component'
        if label_comp not in plotted_legend_labels:
            ax.plot(time_axis[valid_data_mask], component_data[valid_data_mask],
                    color='black', linewidth=1.0, label=label_comp)
            plotted_legend_labels.add(label_comp)
        else: # Plot without label if already added
             ax.plot(time_axis[valid_data_mask], component_data[valid_data_mask],
                    color='black', linewidth=1.0)

        # Plot segment boundaries
        label_bound = 'Segment Boundary'
        for b_idx in segment_boundary_indices:
            if 0 < b_idx < num_waypoints: # Ensure boundary is within plot range
                if label_bound not in plotted_legend_labels:
                    ax.axvline(x=b_idx, color='grey', linestyle='--', linewidth=1.0, label=label_bound)
                    plotted_legend_labels.add(label_bound)
                else: # Plot without label if already added
                    ax.axvline(x=b_idx, color='grey', linestyle='--', linewidth=1.0)

        ax.set_ylabel(feature_name)
        ax.grid(True, linestyle=':', which='both', axis='both')

    # --- Clean Up Unused Axes ---
    for i in range(n_features_to_plot, nrows * ncols):
        if i < len(axs_flat):
            fig.delaxes(axs_flat[i])

    # --- Add X-axis Label to Bottom Row ---
    last_row_plots_indices = range((nrows - 1) * ncols, n_features_to_plot)
    for plot_idx in last_row_plots_indices:
        if plot_idx < len(axs_flat):
            axs_flat[plot_idx].set_xlabel('Waypoint Index')

    # --- Add Combined Legend ---
    if plotted_legend_labels:
        # Get handles and labels from the first subplot (they should be the same)
        handles, labels = axs_flat[0].get_legend_handles_labels()
        # Create a unique dictionary to handle potential duplicates
        by_label = dict(zip(labels, handles))
        # Place legend below the plots
        fig.legend(by_label.values(), by_label.keys(), loc='lower center',
                   bbox_to_anchor=(0.5, -0.05 if nrows > 1 else -0.1), # Adjust vertical position based on rows
                   ncol=min(len(by_label), 6), fontsize='small')

    # Adjust layout to prevent overlap and make space for legend
    plt.tight_layout(rect=[0, 0.05, 1, 0.96]) # rect=[left, bottom, right, top]


class Visualizer3D:
    """
    Creates and manages a 3D plot using Matplotlib for visualizing
    robot trajectories, configurations, and environment elements.
    """
    def __init__(self, title: str = "3D Robot Motion Visualization"):
        """
        Initializes the 3D plotting environment.

        Args:
            title: The title for the 3D plot.
        """
        self.fig = plt.figure(figsize=(10, 8))
        self._Axes3D = Axes3D # Store for potential use, check needed before access

        try:
            # Attempt to add a 3D subplot
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.is_3d = True
            print("INFO (Visualizer3D): Initialized 3D axes.")
        except Exception as e:
            # Fallback to 2D if 3D fails (e.g., missing toolkit)
            print(f"Warning (Visualizer3D): Failed to create 3D axes ({e}). Falling back to 2D.")
            self.ax = self.fig.add_subplot(111)
            self.is_3d = False

        self.ax.set_title(title)
        self.ax.set_xlabel("X (Base Frame)")
        self.ax.set_ylabel("Y (Base Frame)")
        if self.is_3d:
            self.ax.set_zlabel("Z (Base Frame)")

        self.ax.grid(True)

        # --- Set Plot Limits ---
        # Use limits from config if available, otherwise use defaults
        default_limits = ([-0.5, 0.5], [-0.5, 0.5], [0, 1.0])
        limits = getattr(config, 'VIS_PLOT_LIMITS', default_limits)

        # Validate limits format
        if not (isinstance(limits, (list, tuple)) and len(limits) == 3 and
                all(isinstance(lim, (list, tuple)) and len(lim) == 2 for lim in limits)):
            print(f"Warning (Visualizer3D): Invalid VIS_PLOT_LIMITS format ({limits}). Using defaults.")
            limits = default_limits

        self.ax.set_xlim(limits[0])
        self.ax.set_ylim(limits[1])
        if self.is_3d:
            self.ax.set_zlim(limits[2])
            # Set aspect ratio to be equal for 3D plots
            try:
                self.ax.set_box_aspect([np.ptp(lim) for lim in limits]) # ptp = peak-to-peak (range)
            except Exception as e:
                print(f"Warning (Visualizer3D): Could not set box aspect ratio: {e}")

        # --- Plot Base Frame ---
        base_frame_size = getattr(config, 'VIS_BASE_FRAME_SIZE', 0.15)
        self.plot_frame(np.identity(4), size=base_frame_size, label="Base Frame")
        print(f"INFO (Visualizer3D): Base frame plotted with size {base_frame_size}.")


    def plot_frame(self, T: np.ndarray, size: float = 0.1, label: str = ""):
        """
        Plots a 3D coordinate frame represented by a 4x4 homogeneous transform matrix T.

        Args:
            T: 4x4 numpy array representing the frame's pose.
            size: The length of the frame axes.
            label: A label prefix for the frame axes in the legend.
        """
        if not self.is_3d:
            # print("Warning (plot_frame): Cannot plot frame in 2D mode.") # Less verbose
            return
        if T is None or T.shape != (4, 4):
            print(f"Warning (plot_frame): Invalid transform matrix provided for label '{label}'. Skipping.")
            return

        try:
            origin = T[:3, 3]
            x_axis = T[:3, 0]
            y_axis = T[:3, 1]
            z_axis = T[:3, 2]

            colors = ['red', 'green', 'blue']
            axes_vectors = [x_axis, y_axis, z_axis]
            axis_labels = ['X', 'Y', 'Z']

            for i, axis_vec in enumerate(axes_vectors):
                # Add quiver for each axis
                self.ax.quiver(
                    origin[0], origin[1], origin[2], # Start point
                    axis_vec[0], axis_vec[1], axis_vec[2], # Direction vector
                    length=size,
                    color=colors[i],
                    normalize=False, # Use actual vector length for scaling
                    arrow_length_ratio=0.3,
                    # Add label only to the first axis (X) to represent the frame
                    label=f"{label} {axis_labels[i]}" if i == 0 and label else f"_{label}_{axis_labels[i]}" # Underscore hides from legend
                )
        except Exception as e:
            print(f"Warning (plot_frame): Failed to plot frame '{label}': {e}")


    def plot_robot_config(self, q: np.ndarray, color: str = 'grey', linewidth: float = 2, style: str = '-', label: str = "Robot Config"):
        """
        Plots the robot links in task space for a given joint configuration q.
        Requires forward kinematics to be available via the 'utils' module.

        Args:
            q: Numpy array representing the robot joint configuration.
            color: Color of the robot links.
            linewidth: Width of the robot links.
            style: Linestyle for the robot links.
            label: Legend label for this robot configuration.
        """
        if not IK_SOLVER_AVAILABLE:
            # print("Warning (plot_robot_config): IK Solver (for FK) not available. Skipping.") # Less verbose
            return
        if q is None:
            print("Warning (plot_robot_config): Joint configuration 'q' is None. Skipping.")
            return
        # Check if FK function exists in the (potentially dummy) utils module
        if not hasattr(utils, 'forward_kinematics') or not callable(utils.forward_kinematics):
             print("Warning (plot_robot_config): utils.forward_kinematics function not available or not callable. Skipping.")
             return

        try:
            # Calculate joint positions using forward kinematics
            joint_positions, _, _ = utils.forward_kinematics(q)

            if joint_positions is not None:
                jp = np.array(joint_positions)
                # Ensure we have at least two points with 3D coordinates
                if jp.ndim == 2 and jp.shape[0] >= 2 and jp.shape[1] == 3:
                    xs = jp[:, 0]
                    ys = jp[:, 1]
                    if self.is_3d:
                        zs = jp[:, 2]
                        self.ax.plot(xs, ys, zs, color=color, linewidth=linewidth,
                                     linestyle=style, label=label, marker='o', markersize=4)
                    else: # 2D fallback plot
                        self.ax.plot(xs, ys, color=color, linewidth=linewidth,
                                     linestyle=style, label=label, marker='o', markersize=4)
                else:
                    print(f"Warning (plot_robot_config): Invalid joint positions shape received from FK: {jp.shape}. Skipping plot.")
            else:
                print("Warning (plot_robot_config): Forward kinematics returned None for joint positions. Skipping plot.")

        except Exception as e:
            print(f"Warning (plot_robot_config): Failed to plot robot configuration: {e}")
            # traceback.print_exc() # Optional: for more detailed debugging


    def plot_trajectory(
        self,
        path_cspace_np: Optional[np.ndarray],
        num_snapshots: int = 10,
        color: str = 'purple',
        style: str = '-',
        label_prefix: str = "Path",
        plot_tcp_frames: bool = True,
        linewidth: float = 1.5
    ):
        """
        Plots a C-space trajectory in 3D task space by calculating the TCP pose
        at each point using forward kinematics. Optionally plots robot snapshots
        and TCP frames along the path.

        Args:
            path_cspace_np: Numpy array of C-space configurations (shape [N, num_joints]).
            num_snapshots: Number of robot configuration snapshots to plot along the path.
            color: Color for the trajectory line and snapshots.
            style: Linestyle for the trajectory line.
            label_prefix: Prefix for legend labels related to this trajectory.
            plot_tcp_frames: Whether to plot the TCP coordinate frame at snapshot points.
            linewidth: Width of the trajectory line.
        """
        if path_cspace_np is None or len(path_cspace_np) == 0:
            print("Warning (plot_trajectory): No C-space path provided. Skipping.")
            return
        if not IK_SOLVER_AVAILABLE:
            # print("Warning (plot_trajectory): IK Solver (for FK) not available. Skipping.") # Less verbose
            return
        # Check if FK function exists
        if not hasattr(utils, 'forward_kinematics') or not callable(utils.forward_kinematics):
             print("Warning (plot_trajectory): utils.forward_kinematics function not available or not callable. Skipping.")
             return

        tcp_path_taskspace = []
        # Determine indices for snapshots, ensuring start and end are included if possible
        if num_snapshots > 0 and len(path_cspace_np) > 1:
             snapshot_indices = np.linspace(0, len(path_cspace_np) - 1, num_snapshots, dtype=int)
        elif len(path_cspace_np) == 1: # Handle single point trajectory
             snapshot_indices = [0]
        else: # No snapshots if path is empty or num_snapshots is 0
             snapshot_indices = []

        fk_errors = 0
        tcp_frame_size = getattr(config, 'VIS_TCP_FRAME_SIZE', 0.05) # Smaller default for TCP frames

        # Iterate through C-space path to get TCP poses
        for i, q in enumerate(path_cspace_np):
            try:
                # Calculate FK for the current configuration
                joint_positions, T_tcp, _ = utils.forward_kinematics(q)

                if T_tcp is not None and T_tcp.shape == (4, 4):
                    tcp_path_taskspace.append(T_tcp[:3, 3]) # Store TCP position

                    # Plot robot snapshot and/or TCP frame if this index is selected
                    if i in snapshot_indices:
                        # Label only the first snapshot of this trajectory
                        snap_label = f"{label_prefix} Snapshot" if i == snapshot_indices[0] else f"_{label_prefix}_Snapshot_{i}"
                        self.plot_robot_config(q, color=color, linewidth=1, style=':', label=snap_label)

                        if plot_tcp_frames:
                            frame_label = f"{label_prefix} TCP Frame" if i == snapshot_indices[0] else f"_{label_prefix}_TCP_Frame_{i}"
                            self.plot_frame(T_tcp, size=tcp_frame_size, label=frame_label)
                else:
                    fk_errors += 1
                    tcp_path_taskspace.append([np.nan] * 3) # Append NaN if FK failed
                    if T_tcp is None: print(f"Warning (plot_trajectory): FK returned None for T_tcp at index {i}.")
                    else: print(f"Warning (plot_trajectory): FK returned invalid T_tcp shape {T_tcp.shape} at index {i}.")


            except Exception as e:
                fk_errors += 1
                tcp_path_taskspace.append([np.nan] * 3)
                print(f"Warning (plot_trajectory): Exception during FK at index {i}: {e}")

        if fk_errors > 0:
            print(f"Warning (plot_trajectory): Encountered {fk_errors} errors during FK calculations.")

        # --- Plot the TCP Trajectory Line ---
        tcp_path_np = np.array(tcp_path_taskspace)
        valid_tcp_points_mask = ~np.isnan(tcp_path_np).any(axis=1)

        if np.any(valid_tcp_points_mask):
            valid_path = tcp_path_np[valid_tcp_points_mask]
            if len(valid_path) > 0:
                # Use a unique label if no prefix is given
                path_label = label_prefix if label_prefix else f"TCP Path {random.randint(100, 999)}"

                if self.is_3d:
                    self.ax.plot(valid_path[:, 0], valid_path[:, 1], valid_path[:, 2],
                                 color=color, linestyle=style, linewidth=linewidth, label=path_label)
                    # Add markers for start and end points
                    self.ax.scatter(valid_path[0, 0], valid_path[0, 1], valid_path[0, 2],
                                    color=color, marker='>', s=50, label=f"_{path_label}_Start") # Hide with underscore
                    self.ax.scatter(valid_path[-1, 0], valid_path[-1, 1], valid_path[-1, 2],
                                    color=color, marker='s', s=50, label=f"_{path_label}_End") # Use square for end
                else: # 2D fallback plot
                    self.ax.plot(valid_path[:, 0], valid_path[:, 1],
                                 color=color, linestyle=style, linewidth=linewidth, label=path_label)
                    self.ax.scatter(valid_path[0, 0], valid_path[0, 1],
                                    color=color, marker='>', s=50, label=f"_{path_label}_Start")
                    self.ax.scatter(valid_path[-1, 0], valid_path[-1, 1],
                                    color=color, marker='s', s=50, label=f"_{path_label}_End")
            else:
                print(f"Warning (plot_trajectory): No valid TCP points to plot for '{label_prefix}'.")
        else:
            print(f"Warning (plot_trajectory): No valid TCP points found after FK for '{label_prefix}'.")


    def plot_task_space_set(self, set_data: Optional[dict], color: str = 'grey', alpha: float = 0.1, label: str = "Target Set"):
        """
        Plots the 3D bounding box for a task space set defined by position bounds.

        Args:
            set_data: Dictionary possibly containing 'pos_bounds' (shape [3, 2]).
            color: Fill color of the bounding box.
            alpha: Transparency of the bounding box.
            label: Legend label for the bounding box.
        """
        if not self.is_3d:
            # print("Warning (plot_task_space_set): Cannot plot 3D set in 2D mode.") # Less verbose
            return
        if set_data is None or 'pos_bounds' not in set_data:
            print(f"Warning (plot_task_space_set): 'pos_bounds' not found in set_data for label '{label}'. Skipping.")
            return

        bounds = np.array(set_data['pos_bounds'])
        if bounds.shape != (3, 2):
            print(f"Warning (plot_task_space_set): Invalid 'pos_bounds' shape ({bounds.shape}) for label '{label}'. Expected (3, 2). Skipping.")
            return

        xmin, xmax = bounds[0]
        ymin, ymax = bounds[1]
        zmin, zmax = bounds[2]

        # Define the 8 corners of the bounding box
        corners = np.array([
            [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
            [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax]
        ])

        # Define the 6 faces using the corners (order matters for consistent normals)
        faces = [
            [corners[0], corners[1], corners[2], corners[3]], # Bottom face
            [corners[4], corners[5], corners[6], corners[7]], # Top face
            [corners[0], corners[1], corners[5], corners[4]], # Front face
            [corners[2], corners[3], corners[7], corners[6]], # Back face
            [corners[1], corners[2], corners[6], corners[5]], # Right face
            [corners[3], corners[0], corners[4], corners[7]]  # Left face
        ]

        try:
            # Create the 3D polygon collection
            poly3d = Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='k', alpha=alpha)
            self.ax.add_collection3d(poly3d)

            # Create a proxy artist (rectangle) for the legend entry
            proxy = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=alpha, ec='k', label=label)

            # Add the proxy to the legend if the label doesn't exist yet
            current_handles, current_labels = self.ax.get_legend_handles_labels()
            if label not in current_labels:
                 # This approach of modifying the legend directly can be fragile.
                 # Consider using self.add_legend() at the end.
                 # self.ax.legend(current_handles + [proxy], current_labels + [label], loc='best')
                 # For now, just store the proxy to be added later by add_legend
                 if not hasattr(self, '_legend_proxies'): self._legend_proxies = {}
                 self._legend_proxies[label] = proxy
            print(f"INFO (plot_task_space_set): Plotted task space set '{label}'.")

        except NameError:
            print("Warning (plot_task_space_set): Could not plot set - Poly3DCollection not imported/available.")
        except Exception as e:
            print(f"Warning (plot_task_space_set): Failed to plot task space set bounds for '{label}': {e}")


    def plot_tcp_waypoint(self, q: Optional[np.ndarray], color: str = 'red', marker: str = 'o', size: int = 50, label: str = "Waypoint"):
        """
        Plots a marker at the TCP position for a given C-space configuration q.

        Args:
            q: C-space configuration (numpy array).
            color: Marker color.
            marker: Marker style.
            size: Marker size.
            label: Legend label for the waypoint.
        """
        if not IK_SOLVER_AVAILABLE:
            # print("Warning (plot_tcp_waypoint): IK Solver (for FK) not available. Skipping.") # Less verbose
            return
        if q is None:
            print(f"Warning (plot_tcp_waypoint): Joint configuration 'q' is None for label '{label}'. Skipping.")
            return
        # Check if FK function exists
        if not hasattr(utils, 'forward_kinematics') or not callable(utils.forward_kinematics):
             print(f"Warning (plot_tcp_waypoint): utils.forward_kinematics not available for label '{label}'. Skipping.")
             return

        try:
            # Calculate forward kinematics to get TCP transform
            _, T_tcp, _ = utils.forward_kinematics(q)

            if T_tcp is not None and T_tcp.shape == (4,4):
                pos = T_tcp[:3, 3] # Extract position
                if self.is_3d:
                    self.ax.scatter(pos[0], pos[1], pos[2], color=color, marker=marker,
                                    s=size, label=label, depthshade=True) # Enable depthshade for 3D clarity
                else: # 2D fallback
                    self.ax.scatter(pos[0], pos[1], color=color, marker=marker,
                                    s=size, label=label)
            else:
                 print(f"Warning (plot_tcp_waypoint): FK failed or returned invalid shape for label '{label}'. Skipping.")

        except Exception as e:
            print(f"Warning (plot_tcp_waypoint): Failed to plot TCP waypoint for label '{label}': {e}")


    def plot_pso_final_swarm(self, swarm_positions_T: Optional[List[np.ndarray]], best_pos_xyz: Optional[np.ndarray] = None, **kwargs):
        """
        Plots the final positions (task space TCP) of a PSO swarm.

        Args:
            swarm_positions_T: List of 4x4 transform matrices representing particle TCP poses.
            best_pos_xyz: Optional 3D coordinates of the best particle found.
            **kwargs: Additional keyword arguments passed to ax.scatter for swarm points (e.g., s, color, alpha, label).
        """
        if swarm_positions_T is None or len(swarm_positions_T) == 0:
            print("Warning (plot_pso_final_swarm): No swarm positions provided. Skipping.")
            return

        # Extract valid TCP positions (3D) from the list of transforms
        positions = []
        for T in swarm_positions_T:
            if T is not None and T.shape == (4,4):
                positions.append(T[:3, 3])
        positions = np.array(positions)

        if len(positions) == 0:
            print("Warning (plot_pso_final_swarm): No valid TCP positions found in the swarm data. Skipping.")
            return

        # Default scatter plot settings
        size = kwargs.pop('s', 15) # Use pop to remove 's' if present, provide default
        kwargs.setdefault('alpha', 0.6) # Set default alpha if not provided
        kwargs.setdefault('label', 'PSO Particle') # Default label

        print(f"INFO (plot_pso_final_swarm): Plotting {len(positions)} PSO particles.")

        # Plot swarm particles
        if self.is_3d:
            self.ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=size, **kwargs)
            # Plot the best position if provided
            if best_pos_xyz is not None and len(best_pos_xyz) == 3:
                self.ax.scatter(best_pos_xyz[0], best_pos_xyz[1], best_pos_xyz[2],
                                color='lime', marker='*', s=size*6, # Make best marker larger
                                label='PSO Best', depthshade=True, edgecolors='black', zorder=10) # Bring to front
        else: # 2D fallback
            self.ax.scatter(positions[:, 0], positions[:, 1], s=size, **kwargs)
            if best_pos_xyz is not None and len(best_pos_xyz) >= 2:
                 self.ax.scatter(best_pos_xyz[0], best_pos_xyz[1],
                                color='lime', marker='*', s=size*6,
                                label='PSO Best', edgecolors='black', zorder=10)


    def plot_gd_progress(self, pose_history_T: Optional[List[np.ndarray]], cost_history: Optional[List[float]] = None, **kwargs):
        """
        Plots the task space TCP path taken during a Gradient Descent optimization.

        Args:
            pose_history_T: List of 4x4 transform matrices representing TCP poses at each GD step.
            cost_history: Optional list of costs at each step (currently unused in plot).
            **kwargs: Additional keyword arguments passed to ax.plot for the path (e.g., color, linestyle, label).
        """
        if pose_history_T is None or len(pose_history_T) == 0:
            print("Warning (plot_gd_progress): No pose history provided. Skipping.")
            return

        # Extract valid TCP positions
        positions = []
        for T in pose_history_T:
             if T is not None and T.shape == (4,4):
                 positions.append(T[:3, 3])
        positions = np.array(positions)

        if len(positions) == 0:
            print("Warning (plot_gd_progress): No valid TCP positions found in pose history. Skipping.")
            return

        # Default plot settings
        size = kwargs.pop('s', 20) # Base size for markers (start/end)
        marker = kwargs.pop('marker', '>') # Marker along the path line
        markersize = kwargs.pop('markersize', size / 5) # Size of markers along path
        kwargs.setdefault('color', 'cyan')
        kwargs.setdefault('linestyle', '-')
        kwargs.setdefault('alpha', 0.7)
        path_label = kwargs.pop('label', 'GD Path') # Label for the path itself

        print(f"INFO (plot_gd_progress): Plotting {len(positions)} GD steps.")

        # Plot the path line connecting the steps
        if self.is_3d:
            self.ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                         marker=marker, markersize=markersize, label=path_label, **kwargs)
            # Plot start and end markers distinctly
            self.ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                            color=kwargs['color'], marker='o', s=size*2, label='GD Start', zorder=10)
            self.ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                            color='lime', marker='*', s=size*5, label='GD End',
                            depthshade=True, edgecolors='black', zorder=10)
        else: # 2D fallback
            self.ax.plot(positions[:, 0], positions[:, 1],
                         marker=marker, markersize=markersize, label=path_label, **kwargs)
            self.ax.scatter(positions[0, 0], positions[0, 1],
                            color=kwargs['color'], marker='o', s=size*2, label='GD Start', zorder=10)
            self.ax.scatter(positions[-1, 0], positions[-1, 1],
                            color='lime', marker='*', s=size*5, label='GD End',
                            edgecolors='black', zorder=10)


    def add_legend(self, **kwargs):
        """
        Adds a legend to the plot, attempting to filter duplicate labels and
        include any custom proxy artists that were created (e.g., for Poly3DCollection).

        Args:
            **kwargs: Additional keyword arguments passed to ax.legend() (e.g., loc, fontsize).
        """
        handles, labels = self.ax.get_legend_handles_labels()

        # Filter out internal Matplotlib labels (often start with '_')
        filtered_labels_handles = [(l, h) for l, h in zip(labels, handles) if not l.startswith('_')]

        # Use a dictionary to automatically handle duplicate labels (keeps the last one)
        by_label = dict(filtered_labels_handles)

        # Add any stored proxy artists (like the rectangle for Poly3DCollection)
        if hasattr(self, '_legend_proxies'):
            for label, proxy in self._legend_proxies.items():
                if label not in by_label: # Add only if not already present
                    by_label[label] = proxy

        # Only show legend if there are entries
        if by_label:
            # Set default location if not provided
            kwargs.setdefault('loc', 'best')
            self.ax.legend(by_label.values(), by_label.keys(), **kwargs)
            print(f"INFO (add_legend): Added legend with {len(by_label)} unique entries.")
        else:
            print("Warning (add_legend): No valid legend entries found.")


# --- Standalone Plotting Functions ---

def plot_task_space_xyz_components(
    trajectories: Dict[str, Optional[np.ndarray]],
    title: str = "Task Space TCP Position Components (X, Y, Z)"
):
    """
    Plots the X, Y, Z components of one or more task-space TCP trajectories
    against waypoint index/time step. Uses a shared Y-axis range for comparison.

    Args:
        trajectories: Dictionary where keys are labels (str) and values are
                      TCP trajectory numpy arrays (shape [N, 3]) or None.
        title: The main title for the figure.
    """
    print(f"--- Generating Task Space XYZ Component Plot: {title} ---")

    if not trajectories:
        print("  Warning: No trajectories provided to plot_task_space_xyz_components. Skipping.")
        return

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True) # 3 rows, 1 column
    fig.suptitle(title, fontsize=14)

    components = ['X', 'Y', 'Z']
    # Generate distinct colors for each trajectory
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))

    # --- Determine Global Y-axis Limits and Max Length ---
    global_min = float('inf')
    global_max = float('-inf')
    max_len = 0
    valid_trajs_found = False

    for label, traj in trajectories.items():
        if traj is not None and isinstance(traj, np.ndarray) and traj.ndim == 2 and traj.shape[1] == 3:
            # Mask out waypoints with NaN values in any component
            valid_mask = ~np.isnan(traj).any(axis=1)
            if np.any(valid_mask):
                valid_trajs_found = True
                traj_valid = traj[valid_mask]
                global_min = min(global_min, np.min(traj_valid)) # Min across all valid points and components
                global_max = max(global_max, np.max(traj_valid)) # Max across all valid points and components
                max_len = max(max_len, len(traj)) # Max length for x-axis limit
        # else: # Optional: Warn about invalid trajectories
            # print(f"  Warning: Trajectory '{label}' is invalid or None. Skipping for limit calculation.")

    if not valid_trajs_found:
        print("  Warning: No valid trajectory data found to determine plot limits. Using defaults [-1, 1].")
        global_min, global_max = -1.0, 1.0 # Default limits if no data

    # Calculate padding for y-axis
    padding = (global_max - global_min) * 0.05 if global_max > global_min else 0.1
    ylim = (global_min - padding, global_max + padding)

    print(f"  Plotting {len(trajectories)} trajectories. Max length: {max_len}. Y-limits: ({ylim[0]:.3f}, {ylim[1]:.3f})")

    # --- Plot Each Component (X, Y, Z) ---
    for i, comp_name in enumerate(components):
        ax = axs[i]
        plot_count_for_comp = 0
        for k, (label, traj) in enumerate(trajectories.items()):
            # Check again if trajectory is valid before plotting
            if traj is not None and isinstance(traj, np.ndarray) and traj.ndim == 2 and traj.shape[1] == 3:
                waypoints = np.arange(len(traj))
                component_data = traj[:, i]
                valid_mask = ~np.isnan(component_data) # Mask only for this component

                if np.any(valid_mask):
                    ax.plot(waypoints[valid_mask], component_data[valid_mask],
                            label=label, color=colors[k], alpha=0.8)
                    plot_count_for_comp += 1

        ax.set_ylabel(f'{comp_name} (m)')
        ax.grid(True, linestyle=':')
        ax.set_ylim(ylim) # Apply shared y-limits

        # Add legend only to the top subplot
        if i == 0 and plot_count_for_comp > 0:
            ax.legend(loc='best', fontsize='small')
        elif i == 0:
             print("  Warning: No valid data plotted in the first subplot (X), legend skipped.")


    # --- Final Axis Adjustments ---
    if max_len > 0:
        axs[-1].set_xlim(0, max_len - 1) # Set x-axis limit based on longest trajectory
    else:
        axs[-1].set_xlim(0, 1) # Default x-limit if no data

    axs[-1].set_xlabel('Waypoint Index / Time Step')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap


def plot_interface_pose_components_12d(
    target_means_12d_I: Optional[np.ndarray],
    achieved_poses_12d_I: Optional[np.ndarray],
    title: str = "Interface Frame Pose Components (Target vs. Optimized)"
):
    """
    Plots the 12 pose components (tx, ty, tz, r11..r33 flattened rotation matrix)
    in the Interface frame, comparing target means to achieved/optimized poses
    at segment boundaries.

    Args:
        target_means_12d_I: Numpy array of target mean poses (shape [N_boundaries, 12]).
        achieved_poses_12d_I: Numpy array of achieved poses (shape [N_boundaries, 12]).
        title: Title for the plot.
    """
    print(f"--- Generating 12D Interface Pose Component Plot: {title} ---")

    # --- Input Validation ---
    if target_means_12d_I is None or achieved_poses_12d_I is None:
        print("  Warning: Missing target or achieved pose data for 12D plot. Skipping.")
        return
    if not isinstance(target_means_12d_I, np.ndarray) or not isinstance(achieved_poses_12d_I, np.ndarray):
         print("  Warning: Input data must be numpy arrays. Skipping.")
         return
    if target_means_12d_I.shape != achieved_poses_12d_I.shape:
        print(f"  Warning: Shape mismatch between target ({target_means_12d_I.shape}) and achieved ({achieved_poses_12d_I.shape}) poses. Skipping.")
        return
    if target_means_12d_I.ndim != 2 or target_means_12d_I.shape[1] != 12:
        print(f"  Warning: Invalid data shape ({target_means_12d_I.shape}). Expected (N_boundaries, 12). Skipping.")
        return

    num_boundaries = target_means_12d_I.shape[0]
    if num_boundaries == 0:
        print("  Warning: No boundary data provided (0 boundaries). Skipping plot.")
        return

    # Indices for the x-axis (representing segment boundaries)
    boundary_indices = np.arange(1, num_boundaries + 1)

    # Define feature names (3 position + 9 rotation matrix elements)
    feature_names = [
        'tx', 'ty', 'tz',
        'r11', 'r12', 'r13',
        'r21', 'r22', 'r23',
        'r31', 'r32', 'r33'
    ]

    # --- Setup Plot Layout ---
    ncols = 3
    nrows = 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), sharex=True)
    fig.suptitle(title, fontsize=14, y=0.99) # Adjust title position slightly
    axs_flat = axs.flatten() # Flatten for easy iteration

    print(f"  Plotting {num_boundaries} boundaries across 12 components.")

    # --- Plot Each Component ---
    for i, feature_name in enumerate(feature_names):
        if i >= len(axs_flat): break # Should not happen with 4x3 layout for 12 features

        ax = axs_flat[i]

        # Plot target means
        ax.plot(boundary_indices, target_means_12d_I[:, i], marker='x', linestyle='--',
                color='orange', label='Target Mean')

        # Plot achieved/optimized poses
        ax.plot(boundary_indices, achieved_poses_12d_I[:, i], marker='o', linestyle='-',
                color='green', markersize=4, label='Optimized')

        ax.set_ylabel(feature_name)
        ax.grid(True, linestyle=':')

        # Add legend only to the first subplot
        if i == 0:
            ax.legend(loc='best', fontsize='small')

        # Add x-axis label only to the bottom row plots
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel('Boundary Index')

    # --- Final Adjustments ---
    # Set integer ticks for boundary indices if many boundaries
    if num_boundaries > 1:
         axs_flat[-1].set_xticks(boundary_indices) # Ensure integer ticks if space allows

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout


def plot_3d_waypoint_comparison(
    initial_q_trajectory: Optional[np.ndarray], # (N+1) x num_joints C-space waypoints
    optimized_q_trajectory: Optional[np.ndarray], # (N+1) x num_joints C-space waypoints
    title: str = "3D TCP Waypoint Comparison (Initial vs. Optimized)"
):
    """
    Creates a separate 3D plot comparing the initial guess and optimized waypoint
    TCP trajectories by calculating TCP positions via FK and connecting the markers.

    Args:
        initial_q_trajectory: C-space waypoints for the initial guess.
        optimized_q_trajectory: C-space waypoints for the optimized result.
        title: Title for the 3D plot.
    """
    print(f"--- Generating 3D Waypoint Comparison Plot: {title} ---")

    # --- Check Dependencies ---
    if not IK_SOLVER_AVAILABLE:
        print("  Warning: IK Solver (for FK) not available. Skipping 3D waypoint comparison.")
        return
    if not hasattr(utils, 'calculate_tcp_trajectory') or not callable(utils.calculate_tcp_trajectory):
         print("  Warning: utils.calculate_tcp_trajectory function not available. Skipping 3D waypoint comparison.")
         return

    # --- Initialize Plot ---
    fig = plt.figure(figsize=(9, 7))
    try:
        ax = fig.add_subplot(111, projection='3d')
        is_3d = True
    except Exception as e:
        print(f"  Warning: Failed to create 3D axes ({e}). Skipping plot.")
        plt.close(fig) # Close the figure if axes creation failed
        return

    ax.set_title(title)
    ax.set_xlabel("X (Base Frame)")
    ax.set_ylabel("Y (Base Frame)")
    ax.set_zlabel("Z (Base Frame)")
    ax.grid(True)

    plotted_something = False
    all_tcp_points = [] # To collect points for auto-scaling axes

    # --- Process and Plot Initial Trajectory ---
    if initial_q_trajectory is not None:
        initial_tcp_traj = utils.calculate_tcp_trajectory(initial_q_trajectory)
        if initial_tcp_traj is not None and isinstance(initial_tcp_traj, np.ndarray) and initial_tcp_traj.ndim == 2 and initial_tcp_traj.shape[1] == 3:
            valid_mask_init = ~np.isnan(initial_tcp_traj).any(axis=1)
            if np.any(valid_mask_init):
                traj_init = initial_tcp_traj[valid_mask_init]
                if len(traj_init) > 0:
                    print(f"  Plotting initial trajectory ({len(traj_init)} waypoints).")
                    # Plot connecting line (hidden from legend)
                    ax.plot(traj_init[:, 0], traj_init[:, 1], traj_init[:, 2],
                            linestyle='--', color='red', alpha=0.5, label='_Initial Path Connect')
                    # Plot waypoint markers
                    ax.scatter(traj_init[:, 0], traj_init[:, 1], traj_init[:, 2],
                               marker='^', color='red', label='Initial Guess TCP', alpha=0.7, s=35)
                    all_tcp_points.append(traj_init)
                    plotted_something = True
                else: print("  Warning: Initial trajectory had no valid TCP points after FK.")
            else: print("  Warning: Initial trajectory had no valid TCP points (all NaN?).")
        else: print("  Warning: Failed to calculate initial TCP trajectory or result was invalid.")
    else: print("  Info: No initial C-space trajectory provided.")


    # --- Process and Plot Optimized Trajectory ---
    if optimized_q_trajectory is not None:
        optimized_tcp_traj = utils.calculate_tcp_trajectory(optimized_q_trajectory)
        if optimized_tcp_traj is not None and isinstance(optimized_tcp_traj, np.ndarray) and optimized_tcp_traj.ndim == 2 and optimized_tcp_traj.shape[1] == 3:
            valid_mask_opt = ~np.isnan(optimized_tcp_traj).any(axis=1)
            if np.any(valid_mask_opt):
                traj_opt = optimized_tcp_traj[valid_mask_opt]
                if len(traj_opt) > 0:
                    print(f"  Plotting optimized trajectory ({len(traj_opt)} waypoints).")
                    # Plot connecting line (hidden from legend)
                    ax.plot(traj_opt[:, 0], traj_opt[:, 1], traj_opt[:, 2],
                            linestyle='-', color='blue', alpha=0.7, label='_Optimized Path Connect')
                    # Plot waypoint markers
                    ax.scatter(traj_opt[:, 0], traj_opt[:, 1], traj_opt[:, 2],
                               marker='s', color='blue', label='Optimized Waypoint TCP', alpha=0.9, s=40)
                    all_tcp_points.append(traj_opt)
                    plotted_something = True
                else: print("  Warning: Optimized trajectory had no valid TCP points after FK.")
            else: print("  Warning: Optimized trajectory had no valid TCP points (all NaN?).")
        else: print("  Warning: Failed to calculate optimized TCP trajectory or result was invalid.")
    else: print("  Info: No optimized C-space trajectory provided.")

    # --- Adjust Axes and Add Legend ---
    if plotted_something and all_tcp_points:
        # Concatenate all valid points to find bounds
        all_points_np = np.concatenate(all_tcp_points, axis=0)

        if len(all_points_np) > 0:
            min_vals = np.min(all_points_np, axis=0)
            max_vals = np.max(all_points_np, axis=0)
            center = (min_vals + max_vals) / 2.0
            ranges = max_vals - min_vals

            # Handle case where range is zero in some dimension
            ranges[ranges < 1e-6] = 1.0 # Avoid division by zero or tiny ranges

            # Make axes lengths equal based on the largest range + padding
            max_range = np.max(ranges) * 0.6 # 60% of max range for half-width
            if max_range < 0.1: max_range = 0.1 # Ensure a minimum size

            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range, center[1] + max_range)
            ax.set_zlim(center[2] - max_range, center[2] + max_range)

            # Attempt to set equal aspect ratio
            try:
                ax.set_box_aspect([1, 1, 1])
            except Exception as e:
                 print(f"  Warning: Could not set equal box aspect ratio: {e}")

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        # Filter hidden labels
        filtered_hl = [(h, l) for h, l in zip(handles, labels) if not l.startswith('_')]
        if filtered_hl:
            ax.legend([h for h,l in filtered_hl], [l for h,l in filtered_hl], loc='best')
        else:
             print("  Warning: No visible legend items found for 3D comparison plot.")

    elif plotted_something:
         print("  Warning: Plotted data but failed to gather points for axis scaling.")
         ax.legend(loc='best') # Add legend anyway
    else:
        print("  No valid trajectories were plotted. Closing empty figure.")
        plt.close(fig) # Close the figure if nothing was plotted

    # Note: plt.show() is typically called externally after all plots are generated.


def plot_position_covariance_ellipsoid(
    ax: plt.Axes,
    mean_pos: np.ndarray,
    cov_3x3: np.ndarray,
    n_std: float = 1.0,
    color: str = 'grey',
    alpha: float = 0.1,
    resolution: int = 20,
    label: str = ""
) -> List[plt.Line2D]:
    """
    Plots a 3D wireframe ellipsoid representing positional covariance on given 3D axes.
    Requires Scipy for eigenvalue decomposition.

    Args:
        ax: The Matplotlib 3D axes object to plot on.
        mean_pos: The 3D mean position vector (shape [3,]).
        cov_3x3: The 3x3 covariance matrix for position.
        n_std: Number of standard deviations to define the ellipsoid boundary.
        color: Color of the ellipsoid wireframe.
        alpha: Transparency of the wireframe lines.
        resolution: Number of points used to draw the ellipsoid (higher means smoother).
        label: Label for a legend proxy (only if label is non-empty).

    Returns:
        A list of Line2D objects representing the plotted wireframe lines,
        or an empty list if plotting failed or dependencies are missing.
    """
    plotted_lines = [] # Initialize empty list for plotted elements

    # --- Dependency and Input Checks ---
    if not SCIPY_AVAILABLE:
        # print("Warning (plot_position_covariance_ellipsoid): Scipy not available. Cannot plot ellipsoid.") # Less verbose
        return plotted_lines
    if scipy_linalg is None:
         print("Warning (plot_position_covariance_ellipsoid): scipy.linalg not imported. Cannot plot ellipsoid.")
         return plotted_lines
    if mean_pos is None or mean_pos.shape != (3,):
        print(f"Warning (plot_position_covariance_ellipsoid): Invalid mean_pos shape ({mean_pos.shape if mean_pos is not None else 'None'}). Expected (3,). Skipping.")
        return plotted_lines
    if cov_3x3 is None or cov_3x3.shape != (3, 3):
        print(f"Warning (plot_position_covariance_ellipsoid): Invalid cov_3x3 shape ({cov_3x3.shape if cov_3x3 is not None else 'None'}). Expected (3, 3). Skipping.")
        return plotted_lines

    try:
        # --- Calculate Ellipsoid Parameters ---
        # Ensure covariance matrix is symmetric
        cov_3x3_sym = (cov_3x3 + cov_3x3.T) / 2

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = scipy_linalg.eigh(cov_3x3_sym)

        # Check for non-positive eigenvalues (degenerate ellipsoid)
        if np.any(eigenvalues <= 1e-9):
            print(f"Warning (plot_position_covariance_ellipsoid): Non-positive eigenvalues found for '{label}'. Ellipsoid is degenerate. Skipping.")
            # print(f"  Eigenvalues: {eigenvalues}") # Optional debug
            return plotted_lines

        # Radii are sqrt of eigenvalues scaled by n_std
        radii = n_std * np.sqrt(eigenvalues)

        # --- Generate Ellipsoid Points ---
        # Create points on a unit sphere grid
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x_unit = np.outer(np.cos(u), np.sin(v))
        y_unit = np.outer(np.sin(u), np.sin(v))
        z_unit = np.outer(np.ones_like(u), np.cos(v))

        # Scale by radii, rotate by eigenvectors, and translate by mean
        # Stack unit points -> Scale -> Rotate -> Translate
        points_unit_scaled = np.stack((x_unit * radii[0], y_unit * radii[1], z_unit * radii[2]), axis=-1)
        # Reshape for dot product: (resolution, resolution, 3) -> (resolution*resolution, 3)
        points_flattened = points_unit_scaled.reshape(-1, 3)
        # Rotate and translate
        points_transformed = np.dot(points_flattened, eigenvectors.T) + mean_pos
        # Reshape back to grid: (resolution*resolution, 3) -> (resolution, resolution, 3)
        points = points_transformed.reshape(resolution, resolution, 3)


        # --- Plot Wireframe ---
        wireframe_alpha = min(alpha * 2.0, 1.0) # Make wireframe slightly less transparent than fill would be

        # Plot lines along constant v (circles of latitude)
        for i in range(len(v)): # or range(resolution)
            line, = ax.plot(points[i, :, 0], points[i, :, 1], points[i, :, 2],
                            color=color, alpha=wireframe_alpha, linewidth=0.5)
            plotted_lines.append(line)

        # Plot lines along constant u (circles of longitude)
        for i in range(len(u)): # or range(resolution)
            line, = ax.plot(points[:, i, 0], points[:, i, 1], points[:, i, 2],
                            color=color, alpha=wireframe_alpha, linewidth=0.5)
            plotted_lines.append(line)

        # --- Create Legend Proxy (if label provided) ---
        # Note: This proxy isn't automatically added to the legend here.
        # It should be handled by a central legend function like `Visualizer3D.add_legend`.
        if label:
            proxy = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=alpha, label=label)
            # Store it for later use by add_legend (if this function is called within Visualizer3D context)
            # This part might need adjustment depending on how the function is used.
            if hasattr(ax.figure, '_visualizer_instance') and hasattr(ax.figure._visualizer_instance, '_legend_proxies'):
                 ax.figure._visualizer_instance._legend_proxies[label] = proxy
            # Or return the proxy along with lines? For now, just create it.

    except np.linalg.LinAlgError as e:
         print(f"Warning (plot_position_covariance_ellipsoid): Linear algebra error plotting ellipsoid '{label}': {e}")
    except Exception as e:
        print(f"Warning (plot_position_covariance_ellipsoid): Failed to plot ellipsoid '{label}': {e}")
        # traceback.print_exc() # Optional debug

    return plotted_lines
# --- Plot Optimization Component History ---
def plot_optimization_component_history(
    opt_info: Dict[str, Any],
    initial_q_window: List[np.ndarray],
    optimized_q_window: Optional[List[np.ndarray]], # Used for final marker
    boundary_stats: Dict[str, Any], # Used for target line
    H_B_I: np.ndarray, # Base to Interface transform (4x4)
    waypoint_idx_in_window: int = 1, # 1-based index of the VARIABLE waypoint
    component_index: int = 0, # 0=tx, 1=ty, 2=tz
    title_prefix: str = "Optimization Component History"
):
    """
    Plots the value of a specific TCP position component (in Base frame)
    against the optimization iteration number for a single waypoint.

    Args:
        opt_info: Dictionary containing optimization results ('step_history').
        initial_q_window: List of C-space configs *before* optimization.
        optimized_q_window: List of C-space configs *after* optimization (or None).
        boundary_stats: Stats for the target boundary ('mean_pos_I').
        H_B_I: Transform Base to Interface.
        waypoint_idx_in_window: 1-based index of the variable waypoint to plot.
        component_index: Index of the TCP position component (0=tx, 1=ty, 2=tz).
        title_prefix: Prefix for the plot title.
    """
    print(f"--- Generating Optimization Component History Plot: {title_prefix} (Waypoint {waypoint_idx_in_window}, Component {component_index}) ---")

    # --- Basic Checks ---
    if not IK_SOLVER_AVAILABLE or not hasattr(utils, 'calculate_tcp_trajectory'):
        print("  Skipping: Dependencies unavailable (IK Solver or calculate_tcp_trajectory).")
        return

    component_names = ['tx', 'ty', 'tz']
    if not (0 <= component_index < 3):
        print(f"  Error: Invalid component_index ({component_index}). Must be 0, 1, or 2.")
        return
    component_name = component_names[component_index]

    # --- Extract Step History (Similar to plot_optimization_step_details_3d) ---
    step_history_flat = opt_info.get('step_history')
    if not step_history_flat or not isinstance(step_history_flat, (list, np.ndarray)) or len(step_history_flat) == 0:
        print("  Warning: No valid 'step_history' found. Cannot plot.")
        return

    q_path_waypoint_np = None
    try:
        num_joints = len(initial_q_window[0])
        num_elements_per_step = len(step_history_flat[0])
        num_variable_points_in_history = num_elements_per_step // num_joints
        if not (1 <= waypoint_idx_in_window <= num_variable_points_in_history):
             print(f"  Error: waypoint_idx_in_window ({waypoint_idx_in_window}) out of range."); return

        q_idx_in_flat_start = (waypoint_idx_in_window - 1) * num_joints
        q_idx_in_flat_end = q_idx_in_flat_start + num_joints
        q_path_waypoint = [step[q_idx_in_flat_start:q_idx_in_flat_end] for step in step_history_flat if len(step) == num_elements_per_step]

        if not q_path_waypoint: print("  Warning: Could not extract C-space steps."); return
        q_path_waypoint_np = np.array(q_path_waypoint)

    except Exception as e: print(f"  Error processing step history: {e}"); return

    # --- Calculate TCP Trajectory and Extract Component ---
    tcp_path_B = utils.calculate_tcp_trajectory(q_path_waypoint_np)
    if tcp_path_B is None: print("  Warning: Failed to calculate TCP path for history."); return

    component_history = tcp_path_B[:, component_index]
    valid_mask = ~np.isnan(component_history)
    if not np.any(valid_mask): print("  Warning: No valid component values found in history."); return

    iterations = np.arange(len(component_history))

    # --- Get Initial, Final, and Target Values ---
    actual_window_index = waypoint_idx_in_window
    initial_val = np.nan
    final_val = np.nan
    target_val = np.nan

    # Initial
    tcp_initial_B = utils.calculate_tcp_trajectory(np.array([initial_q_window[actual_window_index]]))
    if tcp_initial_B is not None and not np.isnan(tcp_initial_B).any():
        initial_val = tcp_initial_B[0, component_index]

    # Final
    if optimized_q_window is not None and 0 <= actual_window_index < len(optimized_q_window):
        tcp_final_B = utils.calculate_tcp_trajectory(np.array([optimized_q_window[actual_window_index]]))
        if tcp_final_B is not None and not np.isnan(tcp_final_B).any():
            final_val = tcp_final_B[0, component_index]

    # Target
    try:
        if boundary_stats and 'mean_pos_I' in boundary_stats and H_B_I is not None:
            mean_pos_I = np.array(boundary_stats['mean_pos_I'])
            if mean_pos_I.shape == (3,) and H_B_I.shape == (4,4):
                mean_pos_I_h = np.append(mean_pos_I, 1)
                target_mean_pos_B = (H_B_I @ mean_pos_I_h)[:3]
                target_val = target_mean_pos_B[component_index]
    except Exception as e: print(f"  Warning: Could not get target value: {e}")

    # --- Create Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"{title_prefix}\nWaypoint {actual_window_index} (VarIdx {waypoint_idx_in_window}) - Component: {component_name}")
    ax.set_xlabel("Optimization Iteration (Callback Step)")
    ax.set_ylabel(f"{component_name} Value (Base Frame)")
    ax.grid(True, linestyle=':')

    # Plot history
    ax.plot(iterations[valid_mask], component_history[valid_mask], marker='.', linestyle='-', color='blue', label=f'{component_name} History', alpha=0.7)

    # Plot initial value
    if np.isfinite(initial_val):
        ax.scatter(0, initial_val, color='red', marker='o', s=80, label='Initial Value', zorder=5)

    # Plot final value
    if np.isfinite(final_val):
        ax.scatter(len(iterations)-1, final_val, color='green', marker='*', s=120, label='Final Value', zorder=5)
    elif opt_info.get('status') == 'success' and np.any(valid_mask): # Mark last valid point if success but no final value
         last_valid_iter = iterations[valid_mask][-1]
         last_valid_val = component_history[valid_mask][-1]
         ax.scatter(last_valid_iter, last_valid_val, color='cyan', marker='s', s=80, label='Last Valid Step Value', zorder=5)


    # Plot target value
    if np.isfinite(target_val):
        ax.axhline(target_val, color='orange', linestyle='--', linewidth=2, label='Target Mean Value')

    ax.legend(loc='best')
    plt.tight_layout()


# --- Plot Rotation Validation Comparison ---
def plot_rotation_validation_comparison(
    optimized_poses_12d_I: np.ndarray,
    title="Rotation Component Validation (Optimized vs. SVD)"
):
    """
    Compares the 9 rotation components from the raw optimization result
    with the components after SVD validation.

    Args:
        optimized_poses_12d_I: Numpy array of optimized poses (shape [N_boundaries, 12]).
                                The rotation part (indices 3-11) might not be SO(3).
        title: Title for the plot.
    """
    print(f"--- Generating Rotation Validation Comparison Plot: {title} ---")

    # --- Input Validation ---
    if optimized_poses_12d_I is None:
        print("  Warning: Missing optimized pose data for rotation validation plot. Skipping.")
        return
    if not isinstance(optimized_poses_12d_I, np.ndarray) or optimized_poses_12d_I.ndim != 2 or optimized_poses_12d_I.shape[1] != 12:
        print(f"  Warning: Invalid data shape ({optimized_poses_12d_I.shape}). Expected (N_boundaries, 12). Skipping.")
        return

    num_boundaries = optimized_poses_12d_I.shape[0]
    if num_boundaries == 0:
        print("  Warning: No boundary data provided. Skipping plot.")
        return

    # --- Extract and Validate Rotations ---
    original_rot_components = np.zeros((num_boundaries, 9))
    validated_rot_components = np.zeros((num_boundaries, 9))
    svd_failures = 0

    for i in range(num_boundaries):
        # Extract original 9 components and reshape
        R_I_opt_flat = optimized_poses_12d_I[i, 3:]
        original_rot_components[i, :] = R_I_opt_flat
        R_I_opt = R_I_opt_flat.reshape((3, 3))

        # Validate using SVD (assuming utils.validate_rotation_matrix_svd exists)
        if hasattr(utils, 'validate_rotation_matrix_svd'):
            R_I_val = utils.validate_rotation_matrix_svd(R_I_opt)
            if R_I_val is not None:
                validated_rot_components[i, :] = R_I_val.flatten()
            else:
                print(f"  Warning: SVD validation failed for boundary {i}. Using original.")
                validated_rot_components[i, :] = R_I_opt_flat # Fallback
                svd_failures += 1
        else:
            print("  Warning: utils.validate_rotation_matrix_svd not found. Cannot validate.")
            validated_rot_components[i, :] = R_I_opt_flat # Use original if no validation possible
            svd_failures = num_boundaries # Mark all as failed validation

    if svd_failures > 0 and svd_failures < num_boundaries:
         print(f"  Encountered {svd_failures} SVD validation failures.")
    elif svd_failures == num_boundaries:
         print(f"  SVD validation failed or was skipped for all boundaries.")


    # --- Setup Plot Layout (3x3 grid for r11 to r33) ---
    boundary_indices = np.arange(1, num_boundaries + 1)
    rot_feature_names = ['r11','r12','r13','r21','r22','r23','r31','r32','r33']
    ncols = 3
    nrows = 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), sharex=True)
    fig.suptitle(title, fontsize=14, y=0.99)
    axs_flat = axs.flatten()

    print(f"  Plotting {num_boundaries} boundaries across 9 rotation components.")

    # --- Plot Each Rotation Component ---
    for i, feature_name in enumerate(rot_feature_names):
        ax = axs_flat[i]

        # Plot original optimized value
        ax.plot(boundary_indices, original_rot_components[:, i], marker='o', linestyle='-',
                color='blue', markersize=4, alpha=0.7, label='Optimized Raw')

        # Plot SVD validated value
        ax.plot(boundary_indices, validated_rot_components[:, i], marker='x', linestyle='--',
                color='lime', alpha=0.9, label='SVD Validated')

        ax.set_ylabel(feature_name)
        ax.grid(True, linestyle=':')

        # Add legend only to the first subplot
        if i == 0:
            ax.legend(loc='best', fontsize='small')

        # Add x-axis label only to the bottom row plots
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel('Boundary Index')

    # --- Final Adjustments ---
    if num_boundaries > 1:
         axs_flat[-1].set_xticks(boundary_indices)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    
def plot_optimization_step_details_3d(
    opt_info: Dict[str, Any],
    initial_q_window: List[np.ndarray], # Full window [q_fixed, q_var1, ..., q_varN]
    optimized_q_window: Optional[List[np.ndarray]], # Optimized window or None
    boundary_stats: Dict[str, Any], # Stats for the specific boundary being visualized
    H_B_I: np.ndarray, # Base to Interface transform (4x4)
    waypoint_idx_in_window: int = 1, # Index of the VARIABLE waypoint within the window (1=first var point)
    title_prefix: str = "Optimization Steps"
):
    """
    Visualizes the 3D task space path taken during the optimization of a single
    waypoint within a sliding window. Plots the target mean/covariance,
    initial guess, optimized result (if available), and the step-by-step path.

    Args:
        opt_info: Dictionary containing optimization results, including 'step_history'.
        initial_q_window: List of C-space configs in the window *before* optimization.
        optimized_q_window: List of C-space configs in the window *after* optimization, or None if failed.
        boundary_stats: Dictionary with stats for the target boundary ('mean_pos_I', 'covariance_12x12_I').
        H_B_I: 4x4 Homogeneous transform from Base frame to Interface frame.
        waypoint_idx_in_window: 1-based index of the *variable* waypoint within the window
                                 (e.g., 1 for the first point after the fixed one).
        title_prefix: Prefix for the plot title.
    """
    print(f"--- Generating Optimization Step Plot: {title_prefix} (Waypoint Index in Window: {waypoint_idx_in_window}) ---")

    # --- Check Configuration and Dependencies ---
    if not getattr(config, 'VISUALIZE_OPTIMIZATION_STEPS', False):
        print("  Skipping: Optimization step visualization disabled in config (VISUALIZE_OPTIMIZATION_STEPS=False).")
        return
    if not IK_SOLVER_AVAILABLE:
        print("  Skipping: IK Solver (for FK) unavailable.")
        return
    if not SCIPY_AVAILABLE or scipy_linalg is None:
        print("  Skipping: Scipy (for ellipsoid) unavailable.")
        return
    if not hasattr(utils, 'calculate_tcp_trajectory') or not callable(utils.calculate_tcp_trajectory):
         print("  Skipping: utils.calculate_tcp_trajectory function not available.")
         return

    # --- Extract Step History for the Specific Waypoint ---
    step_history_flat = opt_info.get('step_history')
    if not step_history_flat or not isinstance(step_history_flat, (list, np.ndarray)) or len(step_history_flat) == 0:
        print("  Warning: No valid 'step_history' found in opt_info. Cannot plot steps.")
        return

    try:
        # Determine dimensions from initial window and history
        if not initial_q_window or not isinstance(initial_q_window[0], np.ndarray):
             print("  Error: initial_q_window is invalid.")
             return
        num_joints = len(initial_q_window[0])
        num_elements_per_step = len(step_history_flat[0])
        if num_elements_per_step % num_joints != 0:
             print(f"  Error: Step history element size ({num_elements_per_step}) is not a multiple of num_joints ({num_joints}).")
             return
        num_variable_points_in_history = num_elements_per_step // num_joints

        # Validate waypoint index (1-based for variable points)
        if not (1 <= waypoint_idx_in_window <= num_variable_points_in_history):
            print(f"  Error: waypoint_idx_in_window ({waypoint_idx_in_window}) is out of range for the {num_variable_points_in_history} variable points in history.")
            return

        # Calculate start/end indices for the target waypoint's joints in the flattened history
        q_idx_in_flat_start = (waypoint_idx_in_window - 1) * num_joints
        q_idx_in_flat_end = q_idx_in_flat_start + num_joints

        # Extract the C-space path for the specific waypoint
        q_path_waypoint = []
        for q_flat_step in step_history_flat:
            if len(q_flat_step) == num_elements_per_step:
                q_step = q_flat_step[q_idx_in_flat_start:q_idx_in_flat_end]
                if len(q_step) == num_joints:
                    q_path_waypoint.append(q_step)
            else:
                 print(f"  Warning: Skipping step with unexpected length {len(q_flat_step)}.")


        if not q_path_waypoint:
            print("  Warning: Could not extract any valid C-space steps for the specified waypoint.")
            return

        print(f"  Extracted {len(q_path_waypoint)} C-space steps for waypoint {waypoint_idx_in_window}.")
        q_path_waypoint_np = np.array(q_path_waypoint)

    except (IndexError, TypeError, ValueError) as e:
        print(f"  Error processing step history or initial window: {e}")
        return


    # --- Calculate Task Space Paths using FK ---
    # Path of the waypoint during optimization
    tcp_path_B = utils.calculate_tcp_trajectory(q_path_waypoint_np)
    if tcp_path_B is None:
        print("  Warning: Failed to calculate TCP path for optimization steps. Skipping plot.")
        return

    # Initial position of the waypoint
    actual_window_index = waypoint_idx_in_window # The index in the full window (0=fixed, 1=first var, etc.)
    if not (0 <= actual_window_index < len(initial_q_window)):
         print(f"  Error: Waypoint index {actual_window_index} out of bounds for initial_q_window (len {len(initial_q_window)}).")
         return
    tcp_initial_B = utils.calculate_tcp_trajectory(np.array([initial_q_window[actual_window_index]]))
    if tcp_initial_B is None: print(f"  Warning: Failed to calculate initial TCP position for waypoint {actual_window_index}.")

    # Final position of the waypoint (if optimization succeeded and data exists)
    tcp_final_B = None
    if optimized_q_window is not None and 0 <= actual_window_index < len(optimized_q_window):
        tcp_final_B = utils.calculate_tcp_trajectory(np.array([optimized_q_window[actual_window_index]]))
        if tcp_final_B is None: print(f"  Warning: Failed to calculate final TCP position for waypoint {actual_window_index}.")
    elif opt_info.get('status') == 'success':
         print("  Warning: Optimization reported success, but optimized_q_window is missing or index is invalid.")


    # --- Get Target Mean and Covariance in Base Frame ---
    target_mean_pos_B = None
    target_cov_pos_B = None
    try:
        if boundary_stats and 'mean_pos_I' in boundary_stats and 'covariance_12x12_I' in boundary_stats:
            mean_pos_I = np.array(boundary_stats['mean_pos_I']) # Should be (3,)
            cov_12x12_I = np.array(boundary_stats['covariance_12x12_I']) # Should be (12, 12)

            if mean_pos_I.shape == (3,) and cov_12x12_I.shape == (12, 12) and H_B_I is not None and H_B_I.shape == (4,4):
                cov_pos_I = cov_12x12_I[:3, :3] # Extract 3x3 positional covariance in Interface frame

                # Transform mean position to Base frame
                mean_pos_I_h = np.append(mean_pos_I, 1) # Homogeneous coordinates
                target_mean_pos_B = (H_B_I @ mean_pos_I_h)[:3]

                # Transform covariance matrix to Base frame: Cov_B = R * Cov_I * R^T
                R_B_I = H_B_I[:3, :3]
                target_cov_pos_B = R_B_I @ cov_pos_I @ R_B_I.T
                print("  Successfully transformed target mean and covariance to Base frame.")
            else:
                 print("  Warning: Shape mismatch in boundary stats or H_B_I for transformation.")
                 if mean_pos_I.shape != (3,): print(f"    mean_pos_I shape: {mean_pos_I.shape}")
                 if cov_12x12_I.shape != (12, 12): print(f"    cov_12x12_I shape: {cov_12x12_I.shape}")
                 if H_B_I is None or H_B_I.shape != (4,4): print(f"    H_B_I shape: {H_B_I.shape if H_B_I is not None else 'None'}")

        else:
            print("  Warning: Missing 'mean_pos_I' or 'covariance_12x12_I' in boundary_stats.")

    except Exception as e:
        print(f"  Warning: Could not get/transform target mean/covariance: {e}")
        target_mean_pos_B = None
        target_cov_pos_B = None


    # --- Create 3D Plot ---
    fig = plt.figure(figsize=(9, 7))
    try:
        ax = fig.add_subplot(111, projection='3d')
    except Exception as e:
        print(f"  Error creating 3D axes for optimization step plot: {e}. Skipping.")
        plt.close(fig)
        return

    ax.set_title(f"{title_prefix}\nWaypoint {actual_window_index} (Variable Idx {waypoint_idx_in_window}) Position Path")
    ax.set_xlabel("X (Base Frame)")
    ax.set_ylabel("Y (Base Frame)")
    ax.set_zlabel("Z (Base Frame)")
    ax.grid(True)

    plotted_something = False
    legend_handles = []
    all_plot_points = [] # For auto-scaling

    # --- Plot Target Covariance Ellipsoid ---
    if target_mean_pos_B is not None and target_cov_pos_B is not None:
        ell_lines = plot_position_covariance_ellipsoid(
            ax, target_mean_pos_B, target_cov_pos_B,
            n_std=1.0, color='orange', alpha=0.1, label="" # Label handled by proxy
        )
        if ell_lines:
            # Create proxy for legend
            proxy = plt.Rectangle((0, 0), 1, 1, fc='orange', alpha=0.2, label='Target Cov (1 std)')
            legend_handles.append(proxy)
            # Ensure target_mean_pos_B is 2D before appending
            if target_mean_pos_B.ndim == 1:
                all_plot_points.append(target_mean_pos_B.reshape(1, 3)) # Reshape to (1, 3)
            else:
                all_plot_points.append(target_mean_pos_B) # Assume it's already (1, 3) or similar
            plotted_something = True
            print("  Plotted target covariance ellipsoid.")

    # --- Plot Target Mean ---
    if target_mean_pos_B is not None:
        ax.scatter(target_mean_pos_B[0], target_mean_pos_B[1], target_mean_pos_B[2],
                   color='orange', marker='x', s=100, label='Target Mean', depthshade=False, zorder=10)
        # Add to points list only if not already added via ellipsoid and ensure 2D
        if not plotted_something:
             if target_mean_pos_B.ndim == 1:
                all_plot_points.append(target_mean_pos_B.reshape(1, 3)) # Reshape to (1, 3)
             else:
                all_plot_points.append(target_mean_pos_B)
        plotted_something = True
        print("  Plotted target mean.")


    # --- Plot Optimization Path ---
    valid_mask = ~np.isnan(tcp_path_B).any(axis=1)
    if np.any(valid_mask):
        path = tcp_path_B[valid_mask]
        if len(path) > 0:
            # *** MODIFIED STYLE FOR VISIBILITY ***
            ax.plot(path[:, 0], path[:, 1], path[:, 2],
                    linestyle='-', # Solid line
                    color='blue',  # Blue color
                    alpha=0.7,     # Slightly transparent
                    marker='.',    # Keep small markers
                    markersize=5,  # Slightly larger markers
                    label='Optimization Path') # Add label for legend
            all_plot_points.append(path) # path is already 2D (N_steps, 3)
            plotted_something = True
            print(f"  Plotted optimization path ({len(path)} steps).")
        else: print("  Warning: Optimization path had no valid points after masking NaNs.")
    else: print("  Warning: Optimization path contains only NaN values.")


    # --- Plot Initial Guess ---
    if tcp_initial_B is not None and not np.isnan(tcp_initial_B).any():
        pos_init = tcp_initial_B[0] # Should be shape (1, 3) -> (3,)
        ax.scatter(pos_init[0], pos_init[1], pos_init[2], color='red', marker='o',
                   s=80, label='Initial Guess', depthshade=False, zorder=10)
        all_plot_points.append(pos_init.reshape(1,3)) # Reshape to (1,3) for consistency
        plotted_something = True
        print("  Plotted initial guess.")


    # --- Plot Final/Optimized Result ---
    if tcp_final_B is not None and not np.isnan(tcp_final_B).any():
        pos_final = tcp_final_B[0] # Shape (1, 3) -> (3,)
        ax.scatter(pos_final[0], pos_final[1], pos_final[2], color='green', marker='*',
                   s=120, label='Optimized Result', depthshade=False, zorder=10)
        all_plot_points.append(pos_final.reshape(1,3)) # Reshape to (1,3) for consistency
        plotted_something = True
        print("  Plotted optimized result.")
    # If optimization succeeded but final point is missing, plot the last valid step from path
    elif opt_info.get('status') == 'success' and np.any(valid_mask):
         last_valid_step = tcp_path_B[valid_mask][-1]
         ax.scatter(last_valid_step[0], last_valid_step[1], last_valid_step[2],
                    color='blue', marker='s', s=80, label='Opt Last Valid Step',
                    depthshade=False, zorder=10)
         # No need to add to all_plot_points, it's already in the path (which is 2D)
         plotted_something = True
         print("  Plotted last valid step as optimized result placeholder.")


    # --- Final Adjustments: Scaling and Legend ---
    if plotted_something and all_plot_points:
        # Concatenate all points used in the plot for scaling
        # Now all elements in all_plot_points should be 2D arrays
        try:
            all_points_np = np.concatenate(all_plot_points, axis=0)
        except ValueError as e:
             print(f"ERROR during concatenation for axis scaling: {e}")
             print("  Shapes in all_plot_points:")
             for idx, arr in enumerate(all_plot_points):
                 print(f"    Index {idx}: Shape={arr.shape if isinstance(arr, np.ndarray) else type(arr)}")
             # Fallback: Don't scale axes if concatenation fails
             all_points_np = None

        if all_points_np is not None and len(all_points_np) > 0:
            min_vals = np.min(all_points_np, axis=0)
            max_vals = np.max(all_points_np, axis=0)
            center = (min_vals + max_vals) / 2.0
            ranges = max_vals - min_vals
            ranges[ranges < 1e-6] = 0.1 # Ensure minimum range for scaling

            # Make axes lengths equal based on the largest range + padding
            max_range = np.max(ranges) * 0.6 # 60% of max range for half-width
            if max_range < 0.05: max_range = 0.05 # Ensure a minimum size

            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range, center[1] + max_range)
            ax.set_zlim(center[2] - max_range, center[2] + max_range)

            # Attempt to set equal aspect ratio
            try:
                ax.set_box_aspect([1, 1, 1])
            except Exception as e:
                 print(f"  Warning: Could not set equal box aspect ratio for opt step plot: {e}")

        # Add legend (combining proxies and scatter/plot handles)
        handles_plots, labels_plots = ax.get_legend_handles_labels()
        # Filter hidden labels
        valid_handles_labels = [(h, l) for h, l in zip(handles_plots, labels_plots) if not l.startswith('_')]
        # Combine with proxies
        all_handles = legend_handles + [h for h,l in valid_handles_labels]
        all_labels = [h.get_label() for h in legend_handles] + [l for h,l in valid_handles_labels]

        # Create unique legend entries
        by_label = dict(zip(all_labels, all_handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small')
        else:
            print("  Warning: No legend entries generated for optimization step plot.")

    elif plotted_something:
        print("  Warning: Plotted data but failed to gather points for axis scaling in opt step plot.")
        ax.legend(loc='best', fontsize='small') # Add legend anyway
    else:
        print("  No valid data was plotted for optimization steps. Closing empty figure.")
        plt.close(fig) # Close the figure if nothing was plotted

    # Note: plt.show() is typically called externally.


# --- Script Load Confirmation ---
print("visualizer.py loaded and refactored (Complete).")