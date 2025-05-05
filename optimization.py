import numpy as np
import time
import config # Read configuration parameters
import utils # Need utils for transformations, FK, SVD validation, C-space distance
from planner import OMPLPlanner # Need planner for cost evaluation
import traceback # For printing detailed error messages
from typing import Optional, Tuple, List, Dict, Any

# --- Dependency Checks ---
SCIPY_AVAILABLE = utils.SCIPY_AVAILABLE
IK_SOLVER_AVAILABLE = utils.IK_SOLVER_AVAILABLE

# --- SciPy Optimize Import ---
SCIPY_OPTIMIZE_AVAILABLE = False
minimize = None; solve = None
if SCIPY_AVAILABLE:
    try:
        from scipy.optimize import minimize
        from scipy.linalg import solve
        SCIPY_OPTIMIZE_AVAILABLE = True
        print("INFO (optimization.py): Successfully imported scipy.optimize.minimize.")
    except ImportError: print("WARNING (optimization.py): scipy.optimize not found.")
else: print("WARNING (optimization.py): Scipy not available.")


# --- Cost Function for Sliding Window Optimization ---
def sliding_window_cost(
    q_window_flat: np.ndarray, # Flattened array of joint angles for the *variable* points in the window
    q_fixed: np.ndarray, # The fixed starting configuration for this window (q_current)
    boundary_stats_window: List[Dict[str, Any]], # List of stats dicts for boundaries IN the window (len = window_size)
    planner: OMPLPlanner, # Planner instance
    H_I_B: np.ndarray, # Transform Interface to Base (inverse of H_B_I)
    num_variable_points: int, # Number of configurations being optimized (e.g., window_size)
    num_joints: int # Number of joints per configuration
    ) -> float:
    """
    Calculates the combined cost for a set of C-space configurations in a sliding window.
    Cost = w_segment * Cost_Segment + w_pose * Cost_PoseDeviation
    Cost_Segment is either OMPL planning cost or C-space distance based on config flag.
    """
    # ... (Implementation unchanged from immersive id="optimization_py_cost_flag") ...
    if not IK_SOLVER_AVAILABLE: return float('inf')
    if H_I_B is None: return float('inf')
    if len(boundary_stats_window) != num_variable_points: return float('inf')

    total_cost = 0.0
    max_cost_component = 1e12

    try: q_variable_list = q_window_flat.reshape((num_variable_points, num_joints))
    except ValueError: return float('inf')

    q_full_window = [q_fixed] + list(q_variable_list)
    total_segment_cost = 0.0
    num_segments = len(q_full_window) - 1

    for i in range(num_segments):
        q_start = q_full_window[i]; q_goal = q_full_window[i+1]
        segment_cost = max_cost_component
        if np.any(np.isnan(q_start)) or np.any(np.isinf(q_start)) or np.any(np.isnan(q_goal)) or np.any(np.isinf(q_goal)):
             pass # segment_cost remains max_cost_component
        elif config.USE_OMPL_IN_LOCAL_COST:
            planning_time_limit = config.LOCAL_OPT_EVAL_PLANNING_TIME
            try:
                _, path_cost = planner.plan(q_start, q_goal, time_limit=planning_time_limit)
                if path_cost is not None and np.isfinite(path_cost): segment_cost = path_cost
            except Exception: pass
        else:
            try:
                cspace_dist_sq = utils.calculate_cspace_distance_sq_cost(q_start, q_goal)
                if np.isfinite(cspace_dist_sq): segment_cost = cspace_dist_sq
            except Exception: pass
        total_segment_cost += segment_cost
    total_cost += config.LOCAL_OPT_WEIGHT_SEGMENT_COST * total_segment_cost

    total_pose_deviation_cost = 0.0
    reg_covar = 1e-6
    for i in range(num_variable_points):
        q_boundary = q_variable_list[i]; stats = boundary_stats_window[i]
        if np.any(np.isnan(q_boundary)) or np.any(np.isinf(q_boundary)):
            total_pose_deviation_cost += max_cost_component; continue
        try:
            _, T_B_tcp, _ = utils.forward_kinematics(q_boundary)
            if T_B_tcp is None: total_pose_deviation_cost += max_cost_component; continue
            T_I_tcp = H_I_B @ T_B_tcp; p_I_current = T_I_tcp[:3, 3]; R_I_current = T_I_tcp[:3, :3]
            det_R = np.linalg.det(R_I_current)
            # if abs(det_R - 1.0) > 1e-3: print(f"Warning (Cost Func): Det ~ {det_R:.4f}") # Debug
            pose_12D_current = np.concatenate((p_I_current, R_I_current.flatten()))
            mean_12D_target = stats['mean_pose_12D_I']; cov_12x12_target = stats['covariance_12x12_I']
            diff = pose_12D_current - mean_12D_target; cov_regularized = cov_12x12_target + np.identity(12) * reg_covar
            mahalanobis_sq = max_cost_component
            if solve is not None:
                try: y = solve(cov_regularized, diff, assume_a='pos'); mahalanobis_sq = diff.T @ y
                except np.linalg.LinAlgError:
                    try: inv_cov_pseudo = np.linalg.pinv(cov_regularized); mahalanobis_sq = diff.T @ inv_cov_pseudo @ diff
                    except np.linalg.LinAlgError: pass # Use max cost
            if not np.isfinite(mahalanobis_sq): total_pose_deviation_cost += max_cost_component
            else: total_pose_deviation_cost += max(0, mahalanobis_sq)
        except Exception: total_pose_deviation_cost += max_cost_component
    total_cost += config.LOCAL_OPT_WEIGHT_POSE_DIST * total_pose_deviation_cost

    # *** DEBUG PRINT: Cost Component Magnitudes ***
    # print(f"  Debug Cost Components: Segment Cost={total_segment_cost:.6e}, Pose Cost={total_pose_deviation_cost:.6e}")
    # *********************************************

    if not np.isfinite(total_cost): return float('inf')
    return total_cost


# --- Sliding Window Optimization Runner ---
def run_sliding_window_optimization(
    initial_q_window: List[np.ndarray], # Full window [q_fixed, q_var1, q_var2, ...]
    boundary_stats_window: List[Dict[str, Any]], # Stats for the variable points ONLY
    planner: OMPLPlanner,
    H_I_B: np.ndarray
    ) -> Tuple[Optional[List[np.ndarray]], float, Dict]:
    """
    Runs the local optimization (e.g., SLSQP) for a single sliding window.
    Captures intermediate steps via callback if config flag is set.
    """
    opt_info = {
        'status': 'init', 'message': '', 'final_cost': float('inf'),
        'iterations': 0, 'step_history': [] # Add list to store steps
    }

    if not SCIPY_OPTIMIZE_AVAILABLE or minimize is None:
        opt_info['status'] = 'failure'; opt_info['message'] = 'scipy.optimize.minimize not available.'
        print(f"ERROR ({config.LOCAL_OPTIMIZER_TYPE}): {opt_info['message']}")
        return None, float('inf'), opt_info

    if not initial_q_window or len(initial_q_window) < 2:
        opt_info['status'] = 'failure'; opt_info['message'] = 'Initial window must contain at least 2 configurations.'
        print(f"ERROR ({config.LOCAL_OPTIMIZER_TYPE}): {opt_info['message']}")
        return None, float('inf'), opt_info

    q_fixed = initial_q_window[0]; q_variable_initial = initial_q_window[1:]
    num_variable_points = len(q_variable_initial); num_joints = len(q_fixed)

    if len(boundary_stats_window) != num_variable_points:
        opt_info['status'] = 'failure'; opt_info['message'] = f'Mismatch between variable points ({num_variable_points}) and boundary stats ({len(boundary_stats_window)}).'
        print(f"ERROR ({config.LOCAL_OPTIMIZER_TYPE}): {opt_info['message']}")
        return None, float('inf'), opt_info

    q_variable_initial_flat = np.concatenate(q_variable_initial)
    bounds = []
    for _ in range(num_variable_points):
        for j in range(num_joints): bounds.append((config.JOINT_LIMITS_MIN[j], config.JOINT_LIMITS_MAX[j]))

    cost_args = (q_fixed, boundary_stats_window, planner, H_I_B, num_variable_points, num_joints)
    optimizer_method = config.LOCAL_OPTIMIZER_TYPE; max_iter = config.LOCAL_OPT_MAX_ITERATIONS; tolerance = config.LOCAL_OPT_TOLERANCE

    # --- Define Callback Function ---
    iteration_count = [0] # Use list to allow modification within callback
    history = opt_info['step_history'] # Reference the list in opt_info
    # Add initial state to history
    history.append(np.copy(q_variable_initial_flat))


    def optimization_callback(xk, state=None): # <<< MODIFIED LINE
            """Stores intermediate state during optimization."""
            # The 'state' argument captures the extra info passed by some optimizers
            iteration_count[0] += 1
            # Store every N iterations (e.g., every 5) to reduce data size
            # Or store all if needed: if True:
            if iteration_count[0] % 5 == 0:
                history.append(np.copy(xk))
                # print(f"Callback iter {iteration_count[0]}: Stored state.") # Debug
             # print(f"Callback iter {iteration_count[0]}: Stored state.") # Debug

    # --- Call Optimizer with Callback ---
    print(f"--- Running {optimizer_method} Optimization for Window ---")
    print(f"  Num Variable Points: {num_variable_points}")
    print(f"  Segment Cost Type: {'OMPL' if config.USE_OMPL_IN_LOCAL_COST else 'C-Space Distance'}")

    start_time = time.time()
    try:
        callback_func = optimization_callback if config.VISUALIZE_OPTIMIZATION_STEPS else None
        result = minimize(
            sliding_window_cost, q_variable_initial_flat, args=cost_args,
            method=optimizer_method, bounds=bounds, tol=tolerance,
            options={'maxiter': max_iter, 'disp': False},
            callback=callback_func # Pass the callback function
        )
        optimization_time = time.time() - start_time
        print(f"  Optimization finished in {optimization_time:.3f}s. Status: {result.message}")

        opt_info['status'] = 'success' if result.success else 'failure'
        opt_info['message'] = result.message
        opt_info['final_cost'] = result.fun if np.isfinite(result.fun) else float('inf')
        opt_info['iterations'] = result.nit
        opt_info['raw_result'] = result
        # Add final state to history if optimization succeeded
        if result.success:
            history.append(np.copy(result.x))

    except Exception as e:
        optimization_time = time.time() - start_time
        print(f"ERROR ({optimizer_method}): Optimization call failed after {optimization_time:.3f}s: {e}")
        traceback.print_exc(); opt_info['status'] = 'error'; opt_info['message'] = str(e)
        return None, float('inf'), opt_info

    # --- Process Results ---
    if result.success:
        q_variable_optimized_flat = result.x
        try:
            q_variable_optimized = list(q_variable_optimized_flat.reshape((num_variable_points, num_joints)))
            optimized_q_window = [q_fixed] + q_variable_optimized
            print(f"  Optimization Successful. Final Cost: {opt_info['final_cost']:.4f}")
            if config.VISUALIZE_OPTIMIZATION_STEPS: print(f"  Captured {len(history)} optimization steps.")
            return optimized_q_window, opt_info['final_cost'], opt_info
        except ValueError as e:
            print(f"ERROR ({optimizer_method}): Could not reshape optimized result: {e}")
            opt_info['status'] = 'failure'; opt_info['message'] = 'Failed to reshape optimized variables.'
            return None, opt_info['final_cost'], opt_info
    else:
        print(f"  Optimization Failed. Status: {result.message}")
        return None, opt_info['final_cost'], opt_info


# --- REMOVED: Old GMM Optimization Functions ---

print(f"optimization.py loaded (Selectable Cost, Step Capture).")
print(f"Optimization Dependencies: Scipy Optimize Available = {SCIPY_OPTIMIZE_AVAILABLE}, IK Solver Available = {IK_SOLVER_AVAILABLE}")

