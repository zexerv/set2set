import numpy as np
import random
import config # Reads configuration parameters
# Attempt to import from ik_solver, handle if it fails
try:
    # Ensure all necessary functions are imported
    from ik_solver import forward_kinematics, inverse_kinematics, normalize_angles, calculate_jacobian
    IK_SOLVER_AVAILABLE = True
except ImportError:
    print("WARNING (utils.py): Could not import functions from ik_solver.py. Related features will fail.")
    IK_SOLVER_AVAILABLE = False
    # Define dummy functions if ik_solver is missing to avoid NameErrors later
    def forward_kinematics(*args, **kwargs): print("ERROR: forward_kinematics unavailable."); return None, None, None
    def inverse_kinematics(*args, **kwargs): print("ERROR: inverse_kinematics unavailable."); return []
    def normalize_angles(q): print("ERROR: normalize_angles unavailable."); return q # Return input as fallback
    def calculate_jacobian(*args, **kwargs): print("ERROR: calculate_jacobian unavailable."); return None

import traceback # For printing detailed error messages
import yaml # For loading configuration files
from pathlib import Path # For handling file paths
import math # For mathematical operations like sqrt, acos, sin
from typing import Optional, Tuple, List, Dict, Any # Added more specific types

# --- SciPy/TF Import for Transformations ---
try:
    from scipy.spatial.transform import Rotation as R
    SCIPY_AVAILABLE = True
    print("INFO (utils.py): Using scipy.spatial.transform for rotations.")

    def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
        """Converts 3x3 or 4x4 matrix to quaternion [w, x, y, z] using scipy."""
        if not SCIPY_AVAILABLE: raise RuntimeError("Scipy not available.")
        if matrix.shape != (3, 3) and matrix.shape != (4, 4):
            raise ValueError(f"Input matrix must be 3x3 or 4x4, got {matrix.shape}")
        try:
            quat_xyzw = R.from_matrix(matrix[:3, :3]).as_quat()
            return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        except Exception as e:
            print(f"ERROR in matrix_to_quat_wxyz: {e}. Returning identity.")
            traceback.print_exc()
            return np.array([1.0, 0.0, 0.0, 0.0])

    def quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
        """Converts quaternion [w, x, y, z] to 3x3 rotation matrix using scipy."""
        if not SCIPY_AVAILABLE: raise RuntimeError("Scipy not available.")
        quat_wxyz = np.asarray(quat_wxyz)
        if quat_wxyz.shape != (4,): raise ValueError("Input quaternion must have 4 elements.")
        try:
            norm = np.linalg.norm(quat_wxyz)
            if norm < 1e-9: return np.identity(3)
            quat_normalized = quat_wxyz / norm
            quat_xyzw = quat_normalized[[1, 2, 3, 0]]
            return R.from_quat(quat_xyzw).as_matrix()
        except Exception as e:
            print(f"ERROR in quat_wxyz_to_matrix: {e}. Returning identity.")
            traceback.print_exc()
            return np.identity(3)

    def matrix_from_pose_dict(pose_dict: dict) -> Optional[np.ndarray]:
        """Creates 4x4 matrix from pose dict {'position': [x,y,z], 'quaternion': [w,x,y,z]}."""
        if not SCIPY_AVAILABLE: print("ERROR: Scipy required for matrix_from_pose_dict."); return None
        try:
            pos = np.array(pose_dict['position'])
            quat_wxyz = np.array(pose_dict['quaternion']) # Assuming wxyz
            if len(pos) != 3 or len(quat_wxyz) != 4: raise ValueError("Invalid pose dict structure.")
            rot_matrix = quat_wxyz_to_matrix(quat_wxyz)
            matrix = np.identity(4)
            matrix[:3, :3] = rot_matrix
            matrix[:3, 3] = pos
            return matrix
        except (KeyError, ValueError, TypeError) as e:
            print(f"ERROR creating matrix from pose dict: {e}"); return None
        except Exception as e:
            print(f"ERROR creating matrix from pose dict (unexpected): {e}"); traceback.print_exc(); return None

    # --- Orientation Math Utilities using Scipy (Kept for potential visualization/debug) ---
    def log_map_so3(R_mat: np.ndarray) -> Optional[np.ndarray]:
        """Maps SO(3) -> so(3) represented as R^3 vector (axis * angle)."""
        if not SCIPY_AVAILABLE: print("ERROR: Scipy required for log_map_so3."); return None
        if R_mat.shape != (3, 3): raise ValueError("Input must be a 3x3 rotation matrix.")
        try:
            if np.allclose(R_mat, np.identity(3)): return np.zeros(3)
            # Use scipy's as_rotvec which directly computes axis * angle
            rotvec = R.from_matrix(R_mat).as_rotvec()
            return rotvec
        except Exception as e:
            print(f"ERROR calculating log map: {e}"); return None

    def exp_map_so3(v_log: np.ndarray) -> Optional[np.ndarray]:
        """Maps so(3) represented as R^3 vector -> SO(3) (3x3 rotation matrix)."""
        if not SCIPY_AVAILABLE: print("ERROR: Scipy required for exp_map_so3."); return None
        v_log = np.asarray(v_log)
        if v_log.shape != (3,): raise ValueError("Input must be a 3D axis-angle vector.")
        try:
            if np.allclose(v_log, np.zeros(3)): return np.identity(3)
            R_mat = R.from_rotvec(v_log).as_matrix()
            return R_mat
        except Exception as e:
            print(f"ERROR calculating exp map: {e}"); return None

    def quat_to_log_map(q_wxyz: np.ndarray) -> Optional[np.ndarray]:
        """Converts quaternion [w, x, y, z] to log map vector v_log."""
        if not SCIPY_AVAILABLE: print("ERROR: Scipy required for quat_to_log_map."); return None
        try:
            R_mat = quat_wxyz_to_matrix(q_wxyz) # Handles normalization
            if R_mat is None: return None
            return log_map_so3(R_mat)
        except Exception as e: print(f"ERROR in quat_to_log_map: {e}"); return None

    def log_map_to_quat(v_log: np.ndarray) -> Optional[np.ndarray]:
        """Converts log map vector v_log to quaternion [w, x, y, z]."""
        if not SCIPY_AVAILABLE: print("ERROR: Scipy required for log_map_to_quat."); return None
        try:
            R_mat = exp_map_so3(v_log)
            if R_mat is None: return None
            return matrix_to_quat_wxyz(R_mat)
        except Exception as e: print(f"ERROR in log_map_to_quat: {e}"); return None

except ImportError:
    print("CRITICAL WARNING (utils.py): scipy not found. Install with 'pip install scipy'. Transformation functions will fail.")
    SCIPY_AVAILABLE = False
    # Define dummy functions
    def matrix_to_quat_wxyz(matrix): raise NotImplementedError("Scipy required")
    def quat_wxyz_to_matrix(quat_wxyz): raise NotImplementedError("Scipy required")
    def matrix_from_pose_dict(pose_dict): raise NotImplementedError("Scipy required")
    def log_map_so3(R_mat): raise NotImplementedError("Scipy required")
    def exp_map_so3(v_log): raise NotImplementedError("Scipy required")
    def quat_to_log_map(q_wxyz): raise NotImplementedError("Scipy required")
    def log_map_to_quat(v_log): raise NotImplementedError("Scipy required")


print("Loading utils.py...")

# --- NEW: SVD Validation for Rotation Matrix ---
def validate_rotation_matrix_svd(R_matrix: np.ndarray) -> Optional[np.ndarray]:
    """
    Ensures a 3x3 matrix is a valid rotation matrix (in SO(3)) using SVD.
    Returns the closest valid rotation matrix.
    """
    if R_matrix is None or R_matrix.shape != (3, 3):
        print("ERROR (SVD Validate): Input must be a 3x3 numpy array.")
        return None
    try:
        U, _, Vt = np.linalg.svd(R_matrix)
        R_validated = U @ Vt
        # Ensure it's a right-handed system (determinant is +1)
        if np.linalg.det(R_validated) < 0:
            # print("  SVD Validate: Correcting determinant.") # Debug
            Vt[-1, :] *= -1  # Flip the sign of the last row of Vt
            R_validated = U @ Vt
        return R_validated
    except np.linalg.LinAlgError as e:
        print(f"ERROR (SVD Validate): SVD failed: {e}")
        return None
    except Exception as e:
        print(f"ERROR (SVD Validate): Unexpected error: {e}")
        traceback.print_exc()
        return None


# Add this function to your utils.py file

def calculate_tcp_trajectory(q_trajectory: np.ndarray) -> Optional[np.ndarray]:
    """
    Calculates the task-space TCP trajectory (positions only) corresponding
    to a given C-space trajectory using forward kinematics.

    Args:
        q_trajectory: A numpy array of C-space configurations, where each row
                      is a joint configuration (shape [N, num_joints]).

    Returns:
        A numpy array of TCP positions (shape [N, 3]) in the base frame,
        or None if forward kinematics is unavailable or a critical error occurs.
        Rows corresponding to FK failures will contain NaNs.
    """
    print("DEBUG: calculate_tcp_trajectory called.") # Add debug print

    # --- Input Validation ---
    if not isinstance(q_trajectory, np.ndarray) or q_trajectory.ndim != 2:
        print("ERROR (calculate_tcp_trajectory): Input q_trajectory must be a 2D numpy array.")
        return None
    if q_trajectory.shape[0] == 0:
        print("Warning (calculate_tcp_trajectory): Input q_trajectory is empty. Returning empty array.")
        return np.empty((0, 3)) # Return empty array with correct shape
    # Assuming config.NUM_JOINTS is defined elsewhere
    # if q_trajectory.shape[1] != config.NUM_JOINTS:
    #     print(f"ERROR (calculate_tcp_trajectory): Input q_trajectory has incorrect number of joints ({q_trajectory.shape[1]} vs {config.NUM_JOINTS}).")
    #     return None

    # --- Check Dependency ---
    if not IK_SOLVER_AVAILABLE:
        print("ERROR (calculate_tcp_trajectory): IK Solver (required for forward_kinematics) is not available.")
        return None
    if not callable(forward_kinematics):
         print("ERROR (calculate_tcp_trajectory): forward_kinematics function is not available or not callable.")
         return None

    # --- Calculate Trajectory ---
    num_waypoints = q_trajectory.shape[0]
    tcp_positions = []
    fk_failures = 0

    print(f"  Calculating TCP trajectory for {num_waypoints} waypoints...") # Debug

    for i in range(num_waypoints):
        q = q_trajectory[i, :]
        try:
            # Call the existing forward_kinematics function
            # Adjust the unpacking based on the actual return values of your FK function
            # We primarily need the T_tcp matrix (often the second return value)
            _joint_pos, T_tcp, _flange_pose = forward_kinematics(q) # Assumes this signature

            if T_tcp is not None and isinstance(T_tcp, np.ndarray) and T_tcp.shape == (4, 4):
                # Extract the position vector (first 3 elements of the 4th column)
                tcp_pos = T_tcp[:3, 3]
                tcp_positions.append(tcp_pos)
            else:
                # Handle cases where FK returns None or an invalid matrix shape
                print(f"Warning (calculate_tcp_trajectory): FK returned invalid T_tcp for waypoint {i}. Shape: {T_tcp.shape if T_tcp is not None else 'None'}.")
                tcp_positions.append(np.full(3, np.nan)) # Append NaN for this point
                fk_failures += 1

        except Exception as e:
            print(f"ERROR (calculate_tcp_trajectory): Exception during forward_kinematics for waypoint {i}: {e}")
            # traceback.print_exc() # Uncomment for detailed traceback during debugging
            tcp_positions.append(np.full(3, np.nan)) # Append NaN for this point
            fk_failures += 1

    if fk_failures > 0:
        print(f"Warning (calculate_tcp_trajectory): Encountered {fk_failures} FK failures out of {num_waypoints} waypoints.")

    # Convert the list of positions to a NumPy array
    tcp_trajectory_np = np.array(tcp_positions)

    # Final check on the output shape
    if tcp_trajectory_np.shape != (num_waypoints, 3):
         print(f"ERROR (calculate_tcp_trajectory): Unexpected final shape {tcp_trajectory_np.shape}. Expected ({num_waypoints}, 3).")
         # This case indicates a more fundamental issue in the loop logic
         return None

    print(f"  Finished calculating TCP trajectory. Output shape: {tcp_trajectory_np.shape}") # Debug
    return tcp_trajectory_np

# --- YAML Loading Helpers ---
def load_yaml_file(filepath: str) -> Optional[Any]:
    """Loads a YAML file safely, returning None on error."""
    path_obj = Path(filepath)
    if not path_obj.is_file(): print(f"ERROR: YAML file not found: {filepath}"); return None
    try:
        with open(path_obj, 'r') as f: data = yaml.safe_load(f)
        if data is None: print(f"Warning: YAML file is empty: {filepath}"); return {} # Return empty dict/list?
        return data
    except yaml.YAMLError as e: print(f"ERROR parsing YAML file {filepath}: {e}"); return None
    except Exception as e: print(f"ERROR reading file {filepath}: {e}"); return None

def load_interface_transforms(filepath: str) -> Optional[Dict[str, np.ndarray]]:
    """ Loads interface transforms (H_D_I) from YAML."""
    config_data = load_yaml_file(filepath)
    if config_data is None or not isinstance(config_data, dict):
        print(f"ERROR: Invalid format or failed to load interface transforms from {filepath}"); return None
    transforms = {}; loaded_count = 0; error_count = 0
    for interface_id_key, data in config_data.items():
        try:
            if isinstance(data, dict) and 'T_aruco_interface' in data:
                H_D_I = np.array(data['T_aruco_interface'], dtype=float)
                if H_D_I.shape == (4, 4): transforms[interface_id_key] = H_D_I; loaded_count += 1
                else: print(f"Warning: Invalid matrix shape for {interface_id_key}"); error_count += 1
        except Exception as e: print(f"ERROR processing interface {interface_id_key}: {e}"); error_count += 1
    print(f"Loaded {loaded_count} interface transforms from {filepath} ({error_count} errors).")
    if loaded_count == 0 and error_count > 0: return None
    return transforms

def load_environment_config(filepath: str) -> Optional[np.ndarray]:
    """ Loads environment config YAML and extracts H_B_D. """
    config_data = load_yaml_file(filepath)
    if config_data is None or not isinstance(config_data, dict):
        print(f"ERROR: Invalid format or failed to load environment config from {filepath}"); return None
    H_B_D = None
    try:
        device_data = config_data.get('aruco_device', config_data.get('aruco_device_pose'))
        if device_data is None: raise KeyError("Could not find 'aruco_device' key.")
        pose_data = device_data.get('pose')
        if pose_data is None: raise KeyError("Missing 'pose' key.")
        H_B_D = matrix_from_pose_dict(pose_data) # Uses scipy
        if H_B_D is None: raise ValueError("Failed to convert pose dict to matrix.")
        print(f"Loaded device pose (H_B_D) from {filepath}.")
        return H_B_D
    except (KeyError, ValueError, TypeError, NotImplementedError) as e:
        print(f"ERROR extracting device pose from {filepath}: {e}"); return None
    except Exception as e:
        print(f"ERROR: Unexpected error loading environment config {filepath}: {e}"); traceback.print_exc(); return None

# --- NEW: Load Boundary Statistics (12D Pose + Covariance) ---
def load_boundary_statistics(filepath: str) -> Optional[List[Dict[str, Any]]]:
    """
    Loads boundary statistics from YAML, expecting 12D mean pose and 12x12 covariance.
    Validates the mean rotation matrix using SVD.

    Returns:
        List of dictionaries, each containing:
        - 'time_index': Original time index.
        - 'mean_pos_I': 3D numpy array (mean position in Interface frame).
        - 'mean_rot_I_validated': 3x3 numpy array (SVD-validated mean rotation in Interface frame).
        - 'mean_pose_12D_I': 12D numpy array [pos, rot_flat] (original mean from YAML).
        - 'covariance_12x12_I': 12x12 numpy array (covariance in Interface frame).
        - 'raw_data': Original dictionary loaded from YAML for this boundary.
        Or None if loading fails critically.
    """
    print(f"Loading 12D boundary statistics from: {filepath}")
    raw_boundary_list = load_yaml_file(filepath)
    if raw_boundary_list is None or not isinstance(raw_boundary_list, list):
        print("ERROR: Failed to load boundary statistics or format is not a list.")
        return None

    processed_stats = []
    load_errors = 0
    svd_errors = 0

    for i, boundary_data in enumerate(raw_boundary_list):
        try:
            time_index = boundary_data.get('time_index', f'boundary_{i}')
            gmm_params = boundary_data.get('segment_gmm_params')
            if gmm_params is None:
                print(f"Warning: Missing 'segment_gmm_params' for boundary {i} (time: {time_index}). Skipping.")
                load_errors += 1
                continue

            # Expecting single component GMM based on YAML structure provided
            means_list = gmm_params.get('means')
            covariances_list = gmm_params.get('covariances')
            weights_list = gmm_params.get('weights')

            if not means_list or not covariances_list or not weights_list or len(weights_list) != 1:
                print(f"Warning: Invalid GMM structure (expected single component) for boundary {i}. Skipping.")
                load_errors += 1
                continue

            mean_12D_I = np.array(means_list[0], dtype=float)
            cov_12x12_I = np.array(covariances_list[0], dtype=float)

            if mean_12D_I.shape != (12,):
                print(f"Warning: Mean vector shape is {mean_12D_I.shape} (expected 12) for boundary {i}. Skipping.")
                load_errors += 1
                continue
            if cov_12x12_I.shape != (12, 12):
                print(f"Warning: Covariance matrix shape is {cov_12x12_I.shape} (expected 12x12) for boundary {i}. Skipping.")
                load_errors += 1
                continue

            # Extract position and rotation matrix from mean
            p_I_mean = mean_12D_I[:3]
            R_I_mean_flat = mean_12D_I[3:]
            R_I_mean = R_I_mean_flat.reshape((3, 3))

            # Validate the rotation matrix using SVD
            R_I_mean_validated = validate_rotation_matrix_svd(R_I_mean)
            if R_I_mean_validated is None:
                print(f"Warning: SVD validation failed for mean rotation matrix of boundary {i}. Skipping.")
                svd_errors += 1
                load_errors += 1
                continue

            processed_stats.append({
                'time_index': time_index,
                'mean_pos_I': p_I_mean,
                'mean_rot_I_validated': R_I_mean_validated,
                'mean_pose_12D_I': mean_12D_I, # Store original mean
                'covariance_12x12_I': cov_12x12_I,
                'raw_data': boundary_data # Keep original data if needed
            })

        except (KeyError, ValueError, TypeError, IndexError) as e:
            print(f"ERROR processing boundary {i}: Invalid data format - {e}. Skipping.")
            load_errors += 1
        except Exception as e:
            print(f"ERROR processing boundary {i}: Unexpected error - {e}. Skipping.")
            traceback.print_exc()
            load_errors += 1

    print(f"Successfully processed {len(processed_stats)} boundaries.")
    if load_errors > 0:
        print(f"Encountered {load_errors} errors during loading/processing.")
    if svd_errors > 0:
        print(f"Encountered {svd_errors} SVD validation errors.")

    if not processed_stats and load_errors > 0:
        return None # Return None if no boundaries could be processed

    return processed_stats


# --- Robot State Initialization and IK Selection ---
def check_joint_limits(q: np.ndarray) -> bool:
    """Checks if a configuration q is within joint limits defined in config."""
    return np.all(q >= config.JOINT_LIMITS_MIN) and np.all(q <= config.JOINT_LIMITS_MAX)

def calculate_manipulability_cost(q: np.ndarray, epsilon=config.MANIPULABILITY_EPSILON, max_cost=config.MAX_MANIPULABILITY_COST) -> float:
    """ Calculates inverse manipulability cost = 1 / (sqrt(det(JJ^T)) + eps). """
    if not IK_SOLVER_AVAILABLE: return max_cost
    try: J = calculate_jacobian(q)
    except NameError: return max_cost
    if J is None or J.shape[0] != 6 or J.shape[1] != config.NUM_JOINTS: return max_cost
    try: JJT = J @ J.T; det_JJT = np.linalg.det(JJT)
    except np.linalg.LinAlgError: return max_cost
    # Use epsilon for thresholding determinant near zero
    if det_JJT < epsilon**2: # Check against squared epsilon for consistency
        w = 0.0
    else:
        w = math.sqrt(det_JJT) # Determinant is positive
    cost = 1.0 / (w + epsilon); return min(cost, max_cost)
    # except Exception as e: print(f"Warning: Unexpected error during manipulability cost calc: {e}"); return max_cost

def calculate_joint_limit_cost(q: np.ndarray, epsilon=config.IK_SELECT_JLIM_EPSILON) -> float:
    """ Calculates a cost based on proximity to joint limits. """
    cost = 0.0; q_np = np.asarray(q); q_min = config.JOINT_LIMITS_MIN; q_max = config.JOINT_LIMITS_MAX
    dist_to_min = np.abs(q_np - q_min); dist_to_max = np.abs(q_max - q_np)
    # Add small value to denominator to avoid division by zero if exactly at limit
    cost = np.sum(1.0 / (dist_to_min + epsilon)) + np.sum(1.0 / (dist_to_max + epsilon)); return cost

def calculate_cspace_distance_sq_cost(q1: Optional[np.ndarray], q2: Optional[np.ndarray]) -> float:
    """ Calculates squared Euclidean distance between two C-space configurations. """
    if q1 is None or q2 is None: return 0.0 # Or should it be inf if one is None? Depends on usage.
    try:
        q1_np = np.asarray(q1); q2_np = np.asarray(q2)
        if q1_np.shape != q2_np.shape or q1_np.ndim != 1 or q1_np.shape[0] != config.NUM_JOINTS: return float('inf')
        diff = q1_np - q2_np; return np.dot(diff, diff)
    except Exception: return float('inf')

def initialize_robot_cspace(initial_joint_angles: List[float]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[np.ndarray]]]:
    """ Initializes robot from C-space config, performs FK. """
    if not IK_SOLVER_AVAILABLE: print("ERROR: IK Solver (for FK) not available."); return None, None, None
    try:
        q_init_c = np.array(initial_joint_angles, dtype=float)
        if q_init_c.shape != (config.NUM_JOINTS,): return None, None, None
        q_init_c_norm = np.array(normalize_angles(q_init_c))
        if not check_joint_limits(q_init_c_norm): print(f"WARNING: Initial C-space config outside limits.")
        joint_positions, T0_TCP, T0_Flange = forward_kinematics(q_init_c_norm) # Get flange pose too if needed
        if T0_TCP is None: raise ValueError("Forward Kinematics returned None")
        return q_init_c_norm, T0_TCP, joint_positions # Return T0_Flange if needed elsewhere
    except Exception as e: print(f"ERROR during C-space initialization (FK): {e}"); traceback.print_exc(); return None, None, None

# This function selects the 'best' overall IK solution based on cost.
# It might be used for setting an initial robot state from a target pose,
# but is NOT used for the main branch selection or sliding window logic.
def initialize_robot_taskspace(T_target_B: np.ndarray, current_config_hint: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[np.ndarray]]]:
    """ Finds the 'best' C-space config for a target Task Space pose (Base frame) using IK cost evaluation. """
    if not IK_SOLVER_AVAILABLE: print("ERROR: IK Solver not available."); return None, None, None
    print("INFO: Calling initialize_robot_taskspace (selects overall best IK, not branch-specific).")
    try: raw_solutions_info = inverse_kinematics(T_target_B)
    except Exception as e: print(f"ERROR calling IK: {e}"); return None, None, None
    if not raw_solutions_info: return None, None, None

    best_q_so_far = None; min_total_cost = float('inf'); found_valid = False; q_ref = current_config_hint
    for i, sol_info in enumerate(raw_solutions_info):
        try: # Parse solution format (assuming list or tuple with angles first)
            if isinstance(sol_info, (tuple, list)) and len(sol_info) > 0 and isinstance(sol_info[0], (list, np.ndarray)): q_raw = sol_info[0]
            elif isinstance(sol_info, (list, np.ndarray)): q_raw = sol_info
            else: continue
            q_norm = np.array(normalize_angles(q_raw))
            if not check_joint_limits(q_norm): continue
            cost_manip = calculate_manipulability_cost(q_norm)
            cost_jlim = calculate_joint_limit_cost(q_norm)
            cost_dist = calculate_cspace_distance_sq_cost(q_norm, q_ref)
            if not (np.isfinite(cost_manip) and np.isfinite(cost_jlim) and np.isfinite(cost_dist)): continue
            total_cost = (config.IK_SELECT_WEIGHT_MANIP_COST * cost_manip + config.IK_SELECT_WEIGHT_JLIM_COST * cost_jlim + config.IK_SELECT_WEIGHT_DIST_COST * cost_dist)
            if total_cost < min_total_cost: min_total_cost = total_cost; best_q_so_far = q_norm; found_valid = True
        except Exception as e_eval: print(f"ERROR evaluating IK sol {i+1}: {e_eval}")

    if found_valid and best_q_so_far is not None:
        try: # Final FK check
            joint_positions_best, T_final_check, _ = forward_kinematics(best_q_so_far)
            if T_final_check is None: raise ValueError("FK failed for best IK solution")
            # pose_diff = np.linalg.norm(T_final_check[:3,3] - T_target_B[:3,3]) # Optional check
            # if pose_diff > 1e-3: print(f"WARNING: FK vs target pose diff {pose_diff:.4f}m")
            return best_q_so_far, T_final_check, joint_positions_best
        except Exception as e_fk: print(f"ERROR during final FK for best IK: {e_fk}"); return None, None, None
    else: return None, None, None

# *** Helper Function for Branch Following (KEEP AS IS) ***
def get_ik_solution_for_branch(T_target_B: np.ndarray,
                               branch_tuple: Tuple[int, int, int],
                               current_config_hint: Optional[np.ndarray] = None # Hint currently unused but kept
                               ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Finds the specific IK solution corresponding to the given branch tuple.
    Performs FK check and basic validation.
    """
    # --- DEBUG PRINT ---
    # print(f"DEBUG [get_ik_branch]: Target Branch={branch_tuple}, Target Pose Pos={np.round(T_target_B[:3,3], 5)}")
    # --- END DEBUG ---

    if not IK_SOLVER_AVAILABLE: print("ERROR: IK Solver not available."); return None, None
    try:
        # Get all raw solutions and their branch indices
        raw_solutions_info = inverse_kinematics(T_target_B) # Returns list of (angles, t1, t5, t3)
        # --- DEBUG PRINT ---
        # print(f"DEBUG [get_ik_branch]: IK Solver returned {len(raw_solutions_info)} raw solutions.")
        # Optional: Print all raw solutions for comparison across runs
        # for idx, dbg_sol in enumerate(raw_solutions_info):
        #     if isinstance(dbg_sol, (tuple, list)) and len(dbg_sol) == 4:
        #         print(f"  Raw Sol {idx}: Branch={dbg_sol[1:]}, Angles={np.round(np.degrees(dbg_sol[0]), 3)}")
        # --- END DEBUG ---
    except Exception as e:
        print(f"ERROR calling inverse_kinematics in get_ik_solution_for_branch: {e}")
        return None, None

    found_q = None
    # Iterate through solutions to find the one matching the branch tuple
    for sol_idx, sol_info in enumerate(raw_solutions_info):
        try:
            # Check format: expects (angles, t1_idx, t5_idx, t3_idx)
            if isinstance(sol_info, (tuple, list)) and len(sol_info) == 4:
                q_raw, t1_idx, t5_idx, t3_idx = sol_info
                current_branch_tuple = (t1_idx, t5_idx, t3_idx)

                # Check if this solution matches the desired branch
                if current_branch_tuple == branch_tuple:
                    # --- DEBUG PRINT ---
                    # print(f"DEBUG [get_ik_branch]: Found matching raw solution {sol_idx} for branch {branch_tuple}. Raw Angles (deg): {np.round(np.degrees(q_raw), 3)}")
                    # --- END DEBUG ---
                    q_norm = np.array(normalize_angles(q_raw))
                    # Validate joint limits
                    if check_joint_limits(q_norm):
                        found_q = q_norm
                        # --- DEBUG PRINT ---
                        # print(f"DEBUG [get_ik_branch]: Solution passed joint limits. Selected q_norm (deg): {np.round(np.degrees(found_q), 3)}")
                        # --- END DEBUG ---
                        break # Found the matching valid solution
                    # else:
                        # --- DEBUG PRINT ---
                        # print(f"DEBUG [get_ik_branch]: Solution {sol_idx} for branch {branch_tuple} FAILED joint limits.")
                        # --- END DEBUG ---
            # else: print("Warning: Unexpected IK solution format.") # Debug
        except Exception as e_parse:
            print(f"Error parsing IK solution info {sol_idx}: {e_parse}")
            continue # Skip malformed solutions

    # If a valid solution for the branch was found, perform FK check
    if found_q is not None:
        try:
            _, T_branch_check, _ = forward_kinematics(found_q)
            if T_branch_check is None:
                print(f"ERROR: FK failed for selected branch {branch_tuple} solution.")
                return None, None # FK failure is critical

            # Optional: Check if T_branch_check is close to T_target_B
            pos_diff = np.linalg.norm(T_branch_check[:3,3] - T_target_B[:3,3])
            # --- DEBUG PRINT ---
            # print(f"DEBUG [get_ik_branch]: FK Check for branch {branch_tuple}. Result Pos={np.round(T_branch_check[:3,3], 5)}. Pos Diff from Target={pos_diff:.6f}")
            # --- END DEBUG ---
            if pos_diff > 1e-3: # Adjust tolerance as needed
                print(f"WARNING: FK of branch {branch_tuple} solution differs significantly from target pose by {pos_diff:.4f}m")

            return found_q, T_branch_check
        except Exception as e_fk:
            print(f"ERROR during final FK for branch {branch_tuple} solution: {e_fk}")
            return None, None
    else:
        # --- DEBUG PRINT ---
        # print(f"DEBUG [get_ik_branch]: No valid solution found or selected for branch {branch_tuple}.")
        # --- END DEBUG ---
        return None, None # No valid solution found for this branch


# --- REMOVED OBSOLETE FUNCTIONS ---
# - extract_target_pose_from_gmm (replaced by loading 12D stats and SVD)
# - define_set_from_segment_data (part of old traditional approach)
# - sample_pose_from_set (part of old traditional approach)
# - generate_perturbed_config (likely obsolete)

print(f"utils.py loaded (SCIPY_AVAILABLE={SCIPY_AVAILABLE}). Sliding window approach setup.")
