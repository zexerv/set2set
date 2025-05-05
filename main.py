import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import config # Reads updated config (sliding window)
import utils # Reads updated utils (12D stats loading, SVD)
from planner import OMPLPlanner
import visualizer # Import visualizer module
import optimization # Import the new optimization functions
import traceback # Import for better error reporting
import itertools # For generating branch tuples

# --- Import specific IK/FK needed ---
try:
    from ik_solver import forward_kinematics
    IK_SOLVER_AVAILABLE = utils.IK_SOLVER_AVAILABLE
    if not IK_SOLVER_AVAILABLE: raise ImportError("IK Solver functions not available via utils.")
    print("INFO (main.py): Successfully imported FK/IK functions.")
except ImportError as e:
    print(f"FATAL (main.py): Failed to import necessary IK/FK functions: {e}"); sys.exit(1)

# --- Check Optimizer Availability ---
OPTIMIZER_AVAILABLE = optimization.SCIPY_OPTIMIZE_AVAILABLE
if not OPTIMIZER_AVAILABLE: print(f"WARNING (main.py): Optimizer '{config.LOCAL_OPTIMIZER_TYPE}' not available.")

print("Starting main.py execution (Sliding Window Optimization Approach)...")

# --- Helper Function for FK Trajectory ---
def calculate_tcp_trajectory(q_trajectory):
    """Calculates TCP position trajectory using FK."""
    if q_trajectory is None or not IK_SOLVER_AVAILABLE: return None
    tcp_positions = []
    for q in q_trajectory:
        _, T_tcp, _ = utils.forward_kinematics(q)
        if T_tcp is not None: tcp_positions.append(T_tcp[:3, 3])
        else: tcp_positions.append([np.nan, np.nan, np.nan]); # print("Warning: FK failed during TCP trajectory calculation.") # Less verbose
    return np.array(tcp_positions)

# --- Helper Function for 12D Pose Trajectory ---
def calculate_12d_interface_poses(q_trajectory, H_I_B):
    """Calculates 12D poses [pos, rot_flat] in Interface frame."""
    if q_trajectory is None or not IK_SOLVER_AVAILABLE or H_I_B is None: return None
    poses_12d = []
    for q in q_trajectory:
        _, T_B_tcp, _ = utils.forward_kinematics(q)
        if T_B_tcp is not None:
            T_I_tcp = H_I_B @ T_B_tcp; p_I = T_I_tcp[:3, 3]; R_I = T_I_tcp[:3, :3]
            if abs(np.linalg.det(R_I) - 1.0) > 1e-3: print("Warning: Invalid rotation matrix encountered during 12D pose calculation."); poses_12d.append(np.full(12, np.nan))
            else: poses_12d.append(np.concatenate((p_I, R_I.flatten())))
        else: poses_12d.append(np.full(12, np.nan)); # print("Warning: FK failed during 12D pose calculation.") # Less verbose
    return np.array(poses_12d)

# --- Helper Function to Plan and Concatenate Path Segments ---
def plan_and_concatenate_segments(q_waypoints, planner, planning_time, label=""):
    """Plans paths between waypoints and concatenates them."""
    print(f"\nPlanning {label} path segments between {len(q_waypoints)} waypoints...")
    path_segments = []; total_cost = 0.0; path_successful = True
    if len(q_waypoints) < 2: print(f"ERROR ({label}): Need at least two waypoints."); return None, 0.0
    for i in range(len(q_waypoints) - 1):
        q_start_seg = q_waypoints[i]; q_goal_seg = q_waypoints[i+1];
        path_seg, cost_seg = planner.plan(q_start_seg, q_goal_seg, time_limit=planning_time)
        if path_seg is None or not np.isfinite(cost_seg): print(f"  ERROR: {label} planning FAILED for segment {i+1}."); path_successful = False; break
        else: # print(f"    Segment {i+1} OK (Cost: {cost_seg:.4f}, Waypoints: {len(path_seg)})."); # Verbose
             path_segments.append(path_seg); total_cost += cost_seg
    concatenated_path_np = None
    if path_successful and path_segments:
        path_list_to_stack = []
        if path_segments[0] is not None and len(path_segments[0]) > 0: path_list_to_stack.append(path_segments[0])
        for i in range(1, len(path_segments)):
            seg_path = path_segments[i];
            if seg_path is None or len(seg_path) == 0: continue
            if path_list_to_stack:
                prev_last_point = path_list_to_stack[-1][-1]
                if len(seg_path) > 1 and np.allclose(seg_path[0], prev_last_point, atol=1e-6): path_list_to_stack.append(seg_path[1:])
                elif not np.allclose(seg_path[0], prev_last_point, atol=1e-6): path_list_to_stack.append(seg_path)
            else: path_list_to_stack.append(seg_path)
        if path_list_to_stack:
            try: concatenated_path_np = np.vstack(path_list_to_stack); print(f"{label} path generated ({concatenated_path_np.shape[0]} waypoints). Total Cost: {total_cost:.4f}")
            except ValueError as e: print(f"Error concatenating {label} segments: {e}")
        else: print(f"Error concatenating {label} path: No valid segments.")
    else: print(f"{label} path generation skipped or failed.")
    return concatenated_path_np, total_cost

# --- Main Scenario Function ---
def run_sliding_window_scenario():
    """Defines and runs the motion planning scenario using sliding window optimization."""

    # --- 0. Load Common Configurations & Data ---
    print("\n" + "="*40); print("=== 0. Loading Configuration & Boundary Stats ==="); print("="*40)
    H_B_I = None; H_I_B = None; boundary_stats_list = None; global_interface_id = config.INTERFACE_ID
    try:
        interface_transforms = utils.load_interface_transforms(config.INTERFACE_TRANSFORMS_YAML_PATH)
        if interface_transforms is None: raise ValueError("Interface Transforms YAML invalid")
        H_B_D = utils.load_environment_config(config.ENVIRONMENT_YAML_PATH)
        if H_B_D is None: raise ValueError("Environment Config invalid")
        if not global_interface_id: raise ValueError("INTERFACE_ID not set")
        if global_interface_id not in interface_transforms: raise ValueError(f"Interface ID '{global_interface_id}' not found")
        H_D_I = interface_transforms[global_interface_id]; H_B_I = H_B_D @ H_D_I; print("Calculated H_B_I.")
        try: H_I_B = np.linalg.inv(H_B_I); print("Calculated H_I_B.")
        except np.linalg.LinAlgError: print("FATAL: Cannot calculate H_I_B."); H_I_B = None; sys.exit(1)
        if H_B_I is None or H_I_B is None: raise ValueError("H_B_I or H_I_B calculation failed.")
        boundary_stats_list = utils.load_boundary_statistics(config.PROBABILISTIC_STATS_YAML_PATH)
        if boundary_stats_list is None or not boundary_stats_list: raise ValueError("Failed to load or process boundary statistics.")
        num_boundaries = len(boundary_stats_list); print(f"Loaded and processed {num_boundaries} boundaries.")
    except Exception as e: print(f"FATAL: Error during configuration/data loading: {e}"); traceback.print_exc(); sys.exit(1)

    # --- 1. Initialize Robot Start State ---
    print("\n" + "="*40); print("=== 1. Initializing Robot Start State ==="); print("="*40)
    q_init_rad = np.radians([0, -90, 0, -90, 0, 0]); q0, T0_B, _ = utils.initialize_robot_cspace(q_init_rad)
    if q0 is None: print("FATAL: Failed start state initialization."); sys.exit(1)
    print(f"Start Config (q0, deg): {np.round(np.degrees(q0), 1)}")

    # --- 2. Instantiate Planner ---
    print("\n" + "="*40); print("=== 2. Instantiating OMPL Planner ==="); print("="*40)
    ompl_planner = OMPLPlanner();
    if ompl_planner.si is None: print("FATAL: OMPL Planner setup failed."); sys.exit(1)
    print("Planner instantiated successfully.")

    # --- 3. Initial IK Branch Selection ---
    print("\n" + "="*40); print("=== 3. Selecting Initial IK Branch ==="); print("="*40)
    best_initial_branch_tuple = None; q1_for_best_branch = None; min_cost_to_first_mean = float('inf')
    ik_branch_tuples = list(itertools.product([0, 1], repeat=3))
    try:
        stats_b1 = boundary_stats_list[0]; p_I_mean_b1 = stats_b1['mean_pos_I']; R_I_mean_b1_validated = stats_b1['mean_rot_I_validated']
        T_I_mean_b1_validated = np.identity(4); T_I_mean_b1_validated[:3, :3] = R_I_mean_b1_validated; T_I_mean_b1_validated[:3, 3] = p_I_mean_b1
        T_B_mean_b1 = H_B_I @ T_I_mean_b1_validated;
        for branch_tuple in ik_branch_tuples:
            q_b1_branch, T_b1_check = utils.get_ik_solution_for_branch(T_B_mean_b1, branch_tuple, q0)
            if q_b1_branch is not None:
                planning_time = config.LOCAL_OPT_EVAL_PLANNING_TIME
                path_seg_0_1, cost_0_1 = ompl_planner.plan(q0, q_b1_branch, time_limit=planning_time)
                if path_seg_0_1 is not None and np.isfinite(cost_0_1):
                    if cost_0_1 < min_cost_to_first_mean:
                        min_cost_to_first_mean = cost_0_1; best_initial_branch_tuple = branch_tuple; q1_for_best_branch = q_b1_branch
                        print(f"    *** New Best Branch Found: {best_initial_branch_tuple} (Cost: {min_cost_to_first_mean:.4f}) ***")
    except Exception as e: print(f"FATAL: Error during initial IK branch selection: {e}"); traceback.print_exc(); sys.exit(1)
    if best_initial_branch_tuple is None: print("FATAL: Could not find any valid IK branch."); sys.exit(1)
    print(f"\nSelected IK Branch for Optimization: {best_initial_branch_tuple}")

    # --- 4. Generate Initial Full Trajectory Guess (Based on Means) ---
    print("\n" + "="*40); print("=== 4. Generating Initial Trajectory Guess (Means) ==="); print("="*40)
    initial_q_trajectory = [q0, q1_for_best_branch]; previous_q = q1_for_best_branch; initial_guess_successful = True
    for i in range(1, num_boundaries):
        stats_bN = boundary_stats_list[i]; p_I_mean_bN = stats_bN['mean_pos_I']; R_I_mean_bN_validated = stats_bN['mean_rot_I_validated']
        T_I_mean_bN_validated = np.identity(4); T_I_mean_bN_validated[:3, :3] = R_I_mean_bN_validated; T_I_mean_bN_validated[:3, 3] = p_I_mean_bN
        T_B_mean_bN = H_B_I @ T_I_mean_bN_validated
        q_bN_initial, _ = utils.get_ik_solution_for_branch(T_B_mean_bN, best_initial_branch_tuple, previous_q)
        if q_bN_initial is None: print(f"  ERROR: Failed IK for Boundary {i+1}."); initial_guess_successful = False; break
        initial_q_trajectory.append(q_bN_initial); previous_q = q_bN_initial
    if not initial_guess_successful: print("FATAL: Could not generate full initial trajectory guess."); sys.exit(1)
    print(f"Initial Trajectory Guess (len={len(initial_q_trajectory)}) generated successfully.")
    try: np.save("initial_guess_trajectory.npy", np.array(initial_q_trajectory)); print("Saved initial guess trajectory.")
    except Exception as e: print(f"ERROR saving initial guess trajectory: {e}")

    # --- 4b. Plan Path Based on Initial Guess ---
    initial_planned_path_np, _ = plan_and_concatenate_segments(
        initial_q_trajectory, ompl_planner, config.FINAL_SEGMENT_PLANNING_TIME, "Initial Guess"
    )
    # --- 5. Sliding Window Optimization Loop ---
    print("\n" + "="*40); print("=== 5. Running Sliding Window Optimization ==="); print("="*40)
    optimized_q_trajectory = list(initial_q_trajectory) # Start with the initial guess
    window_size = config.SLIDING_WINDOW_SIZE
    all_optimization_infos = [] # Store info from ALL windows across ALL passes
    num_passes = config.NUM_OPTIMIZATION_PASSES

    if not OPTIMIZER_AVAILABLE:
        print("Skipping optimization loop: Optimizer not available.")
    elif window_size <= 0 or window_size > num_boundaries:
        print(f"Warning: Invalid window size ({window_size}). Skipping optimization.")
    elif num_passes <= 0:
        print(f"Warning: NUM_OPTIMIZATION_PASSES ({num_passes}) is zero or negative. Skipping optimization.")
    else:
        print(f"Starting optimization with {num_passes} pass(es) over the trajectory.")
        # --- Outer Loop for Multiple Passes ---
        for pass_num in range(num_passes):
            print(f"\n===== Optimization Pass {pass_num + 1} / {num_passes} =====")
            # Use the result of the previous pass (or initial guess) as the starting point
            current_q_trajectory = list(optimized_q_trajectory) # Make a copy to modify in this pass
            pass_start_time = time.time()

            # --- Inner Loop for Sliding Window (Existing Loop) ---
            num_opt_steps = num_boundaries # Number of windows to process in one pass
            for i in range(num_opt_steps):
                window_start_time = time.time()
                fixed_q_index = i
                start_variable_index = i + 1
                # Ensure the window doesn't go beyond the trajectory length
                end_variable_index = min(i + 1 + window_size, num_boundaries + 1)
                variable_indices = range(start_variable_index, end_variable_index)
                actual_window_size = len(variable_indices)

                if actual_window_size < 1:
                    print(f"  Window {i+1}: No variable points left. Stopping pass.")
                    break # Stop this pass if window is empty

                print(f"\n-- Pass {pass_num+1}, Optimizing Window {i+1} (Fixed q{fixed_q_index}, Optimizing q{list(variable_indices)}) --")

                # --- Prepare data for the current window ---
                # Get configurations from the *current state* of the trajectory for this pass
                current_fixed_q = current_q_trajectory[fixed_q_index]
                current_initial_q_variable = [current_q_trajectory[k] for k in variable_indices]
                current_initial_q_window = [current_fixed_q] + current_initial_q_variable
                # Get corresponding boundary stats (indices are offset by -1 from trajectory indices)
                current_boundary_stats = [boundary_stats_list[k-1] for k in variable_indices]

                # --- Run Optimization for the window ---
                opt_result_window, final_cost, opt_info = optimization.run_sliding_window_optimization(
                    current_initial_q_window,
                    current_boundary_stats,
                    ompl_planner,
                    H_I_B
                )

                # --- Store detailed info (including pass number) ---
                all_optimization_infos.append({
                    'pass_num': pass_num + 1, # Add pass number
                    'window_index': i,
                    'fixed_q_index': fixed_q_index,
                    'variable_indices': list(variable_indices),
                    'initial_window': current_initial_q_window,
                    'optimized_window': opt_result_window, # Store optimized window result
                    'boundary_stats': current_boundary_stats, # Store stats used
                    'opt_info': opt_info # Original info dict from optimizer run
                })

                # --- Update the current trajectory for the *next* window in this pass ---
                if opt_result_window is not None and opt_info.get('status') == 'success':
                    # Update the variable points in current_q_trajectory with the results
                    for k_win, q_opt in enumerate(opt_result_window[1:]):
                        traj_idx = variable_indices[k_win]
                        current_q_trajectory[traj_idx] = q_opt
                    print(f"  Window {i+1} optimization successful. Updated trajectory points {list(variable_indices)}.")
                else:
                    # If optimization failed, keep the points from the start of this pass
                    print(f"  WARNING: Optimization failed for window {i+1} in pass {pass_num+1}. Using trajectory from start of pass.")
                    # No update needed, current_q_trajectory already holds the values

                print(f"  Window {i+1} processing time: {time.time() - window_start_time:.3f}s")
            # --- End of Inner Loop (Sliding Window) ---

            # After processing all windows in a pass, update the main optimized_q_trajectory
            # This becomes the input for the next pass
            optimized_q_trajectory = list(current_q_trajectory)
            print(f"===== Pass {pass_num + 1} finished in {time.time() - pass_start_time:.3f}s =====")
        # --- End of Outer Loop (Passes) ---

    print(f"\nSliding window optimization ({num_passes} passes) complete. Final optimized trajectory length: {len(optimized_q_trajectory)}")
    if len(optimized_q_trajectory) != num_boundaries + 1: print(f"WARNING: Final optimized trajectory length mismatch.")
    try: np.save("optimized_waypoints.npy", np.array(optimized_q_trajectory)); print("Saved optimized waypoints trajectory.")
    except Exception as e: print(f"ERROR saving optimized waypoints trajectory: {e}")

    # --- 6. Final Path Generation (Planning between optimized waypoints) ---
    print("\n" + "="*40); print("=== 6. Generating Final Smooth Path (Optimized) ==="); print("="*40)
    final_refined_path_np, _ = plan_and_concatenate_segments(
        optimized_q_trajectory, ompl_planner, config.FINAL_SEGMENT_PLANNING_TIME, "Final Optimized"
    )

    # --- 7. Calculate Data for Visualization ---
    print("\n" + "="*40); print("=== 7. Calculating Data for Visualization ==="); print("="*40)
    initial_planned_tcp_traj = calculate_tcp_trajectory(initial_planned_path_np)
    final_planned_tcp_traj = calculate_tcp_trajectory(final_refined_path_np)
    initial_waypoints_tcp_traj = calculate_tcp_trajectory(np.array(initial_q_trajectory))
    optimized_waypoints_tcp_traj = calculate_tcp_trajectory(np.array(optimized_q_trajectory))
    optimized_poses_12d_I = calculate_12d_interface_poses(np.array(optimized_q_trajectory[1:]), H_I_B)
    target_means_12d_I = None
    if boundary_stats_list:
        try: target_means_12d_I = np.array([stats['mean_pose_12D_I'] for stats in boundary_stats_list])
        except KeyError: print("Error extracting target means for 12D plot.")

    # --- 8. Visualization ---
    print("\n" + "="*40); print("=== 8. Visualization ==="); print("="*40)
    # *** ADD DEBUG PRINT FOR VISUALIZER PATH ***
    try:
        print(f"DEBUG: Importing visualizer from: {visualizer.__file__}")
    except AttributeError:
        print("DEBUG: Cannot determine visualizer file path.")
    # ******************************************
    vis_title_suffix = f"(SlidingWindow_W{window_size})"
    # Main 3D Animation Visualization (shows final path)
    vis3d = visualizer.Visualizer3D(f"Robot Motion {vis_title_suffix}") # This is the line that might fail (approx line 220)
    vis3d.plot_robot_config(q0, color='black', linewidth=3, style='-', label='Initial Pose (q0)')
    if final_refined_path_np is not None:
        vis3d.plot_trajectory(final_refined_path_np, num_snapshots=config.VIS_3D_SNAPSHOTS, color='green', style='-', label_prefix="Final Path", plot_tcp_frames=True, linewidth=2.0)
        vis3d.plot_robot_config(final_refined_path_np[-1], color='purple', linewidth=2, style='-', label='Final Config')
    vis3d.add_legend(loc='best')

    # Call XYZ component plot (using waypoint trajectories)
    xyz_trajs_to_plot = {}
    if initial_waypoints_tcp_traj is not None: xyz_trajs_to_plot['Initial Guess Wpts'] = initial_waypoints_tcp_traj
    if optimized_waypoints_tcp_traj is not None: xyz_trajs_to_plot['Optimized Waypoints'] = optimized_waypoints_tcp_traj
    if xyz_trajs_to_plot:
        visualizer.plot_task_space_xyz_components(xyz_trajs_to_plot, title=f"Task Space TCP Position {vis_title_suffix}")

    # Call 12D component plot (Target vs Optimized Raw)
    if target_means_12d_I is not None and optimized_poses_12d_I is not None:
        if target_means_12d_I.shape[0] == optimized_poses_12d_I.shape[0]:
            visualizer.plot_interface_pose_components_12d(
                target_means_12d_I,
                optimized_poses_12d_I, # Plotting RAW optimized result here
                title=f"Interface Pose Components {vis_title_suffix}"
            )
        else: print(f"Warning: Skipping 12D plot due to shape mismatch.")

    # Call Rotation Validation Comparison Plot (Optimized Raw vs SVD) (NEW)
    if config.VISUALIZE_ROTATION_VALIDATION and optimized_poses_12d_I is not None:
        visualizer.plot_rotation_validation_comparison(
            optimized_poses_12d_I,
            title=f"Rotation Validation {vis_title_suffix}"
        )
        
    # Call MODIFIED 3D Comparison Plot (using WAYPOINT trajectories)
    if initial_q_trajectory is not None and optimized_q_trajectory is not None:
        visualizer.plot_3d_waypoint_comparison(np.array(initial_q_trajectory), np.array(optimized_q_trajectory), title=f"3D Waypoint Comparison {vis_title_suffix}")
    else:
        print("Warning: Skipping 3D waypoint comparison plot as initial or optimized waypoints are missing.")

    # Call NEW Optimization Step Visualization
    if config.VISUALIZE_OPTIMIZATION_STEPS and all_optimization_infos:
        print("\nGenerating optimization step visualizations...")
        for window_info in all_optimization_infos:
             try:
                 if window_info['opt_info'] and window_info['initial_window'] and window_info['boundary_stats']:
                     visualizer.plot_optimization_step_details_3d(
                         opt_info=window_info['opt_info'],
                         initial_q_window=window_info['initial_window'],
                         optimized_q_window=window_info['optimized_window'],
                         boundary_stats=window_info['boundary_stats'][0],
                         H_B_I=H_B_I,
                         waypoint_idx_in_window=1,
                         title_prefix=f"Opt Window {window_info['window_index']+1}"
                     )
                 else: print(f"Warning: Skipping step visualization for window {window_info['window_index']+1} due to missing data.")
             except Exception as e_vis: print(f"ERROR generating step visualization for window {window_info['window_index']+1}: {e_vis}")

    if plt.get_fignums(): print("\nDisplaying all plots... Close plot windows to exit."); plt.show()
    else: print("No plots were generated.")



    # Call NEW Optimization Step Visualization
    if config.VISUALIZE_OPTIMIZATION_STEPS and all_optimization_infos:
        print("\nGenerating optimization step visualizations...")
        for window_info in all_optimization_infos:
            try:
                if window_info['opt_info'] and window_info['initial_window'] and window_info['boundary_stats']:
                    # --- Plot 3D Step Details ---
                    visualizer.plot_optimization_step_details_3d(
                        opt_info=window_info['opt_info'],
                        initial_q_window=window_info['initial_window'],
                        optimized_q_window=window_info['optimized_window'],
                        boundary_stats=window_info['boundary_stats'][0], # Assuming stats for first variable point
                        H_B_I=H_B_I,
                        waypoint_idx_in_window=1, # Plotting first variable point
                        title_prefix=f"Opt Window {window_info['window_index']+1}"
                    )

                    # --- Plot Component History (NEW) ---
                    if config.VISUALIZE_OPTIMIZATION_COMPONENT_HISTORY:
                        visualizer.plot_optimization_component_history(
                            opt_info=window_info['opt_info'],
                            initial_q_window=window_info['initial_window'],
                            optimized_q_window=window_info['optimized_window'],
                            boundary_stats=window_info['boundary_stats'][0], # Assuming stats for first variable point
                            H_B_I=H_B_I,
                            waypoint_idx_in_window=1, # Plotting first variable point
                            component_index=config.VISUALIZE_COMPONENT_INDEX, # Use index from config
                            title_prefix=f"Opt Window {window_info['window_index']+1}"
                        )

                else: print(f"Warning: Skipping step visualization for window {window_info['window_index']+1} due to missing data.")
            except Exception as e_vis: print(f"ERROR generating step visualization for window {window_info['window_index']+1}: {e_vis}")

    if plt.get_fignums(): print("\nDisplaying all plots... Close plot windows to exit."); plt.show()
    else: print("No plots were generated.")
    
    
    
    # --- 9. Save Final Trajectory ---
    print("\n" + "="*40); print("=== 9. Saving Final Trajectory ==="); print("="*40)
    if final_refined_path_np is not None:
        trajectory_filename = "planned_trajectory_sw.npy"
        try: np.save(trajectory_filename, final_refined_path_np); print(f"Saved final trajectory to: {trajectory_filename}")
        except Exception as e: print(f"ERROR saving trajectory: {e}")
    else: print("No final trajectory generated to save.")

    print("\nmain.py finished.")

# --- Entry Point ---
if __name__ == "__main__":
    if not utils.SCIPY_AVAILABLE: print("\nWARNING: Scipy not found.")
    if not utils.IK_SOLVER_AVAILABLE: print("\nFATAL: IK Solver functions required."); sys.exit(1)
    if not optimization.SCIPY_OPTIMIZE_AVAILABLE: print(f"\nWARNING: Optimizer '{config.LOCAL_OPTIMIZER_TYPE}' unavailable.")
    run_sliding_window_scenario()
