#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone Robot Trajectory Animator from .npy File with Static Frames

Loads a final C-space trajectory, initial guess trajectory, optimized waypoints,
and boundary statistics. Creates an animated 3D plot showing the robot's motion,
its TCP coordinate frame, static Device/Interface frames, boundary means/ellipsoids,
initial guess points, and optimized points.

Usage:
    python animate.py <path_to_final_trajectory.npy> [path_to_initial_guess.npy] [path_to_optimized_waypoints.npy] [config_dir] [num_interpolated_points]

Arguments:
    path_to_final_trajectory.npy: Path to the final planned .npy file (Nx6 C-space trajectory in radians).
    path_to_initial_guess.npy (optional): Path to the initial guess .npy file ( (NumBoundaries+1)x6 ). Defaults to 'initial_guess_trajectory.npy'.
    path_to_optimized_waypoints.npy (optional): Path to the optimized waypoints .npy file ( (NumBoundaries+1)x6 ). Defaults to 'optimized_waypoints.npy'.
    config_dir (optional): Path to the directory containing config.py and YAML files. Defaults to '.' (the current directory).
    num_interpolated_points (optional): Total animation frames (default: 200). Min: 2.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from matplotlib.lines import Line2D # For legend proxies
import math
import argparse
from pathlib import Path
import yaml
import importlib.util # To load config.py dynamically
import traceback
from typing import Optional, List, Dict, Any

# --- SciPy Check (Required for Robust Transformations & Ellipsoids) ---
try:
    from scipy.spatial.transform import Rotation as R
    import scipy.linalg # For ellipsoid calculation
    SCIPY_AVAILABLE = True
    print("INFO: Using scipy.spatial.transform for rotations and scipy.linalg.")

    def quat_wxyz_to_matrix(quat_wxyz):
        """Converts quaternion [w, x, y, z] to 3x3 rotation matrix using scipy."""
        try:
            quat_wxyz = np.asarray(quat_wxyz)
            norm = np.linalg.norm(quat_wxyz)
            if norm < 1e-9: return np.identity(3)
            quat_normalized = quat_wxyz / norm
            quat_xyzw = quat_normalized[[1, 2, 3, 0]]
            return R.from_quat(quat_xyzw).as_matrix()
        except Exception as e:
            print(f"ERROR in quat_wxyz_to_matrix: {e}")
            return np.identity(3)

    def matrix_from_pose_dict(pose_dict):
        """ Creates 4x4 matrix from pose dict {position: [x,y,z], quaternion: [w,x,y,z]}. """
        pos = np.array(pose_dict['position'])
        quat_raw = np.array(pose_dict['quaternion'])
        if len(pos) != 3: raise ValueError("Position must have 3 elements")
        if len(quat_raw) != 4: raise ValueError("Quaternion must have 4 elements")
        quat_wxyz = quat_raw
        rot_matrix = quat_wxyz_to_matrix(quat_wxyz)
        matrix = np.identity(4); matrix[:3, :3] = rot_matrix; matrix[:3, 3] = pos
        return matrix

except ImportError:
    print("CRITICAL ERROR: scipy (including linalg) not found. Install with 'pip install scipy'.")
    SCIPY_AVAILABLE = False
    def quat_wxyz_to_matrix(quat_wxyz): raise NotImplementedError("Scipy required")
    def matrix_from_pose_dict(pose_dict): raise NotImplementedError("Scipy required")
    def plot_position_covariance_ellipsoid(*args, **kwargs): print("Cannot plot ellipsoid: Scipy required.")


# --- Robot Parameters and Forward Kinematics (Keep as is) ---
DH_PARAMS_UR5E = [
    {'a': 0.0,    'alpha': math.pi/2,  'd': 0.1625, 'theta_offset': 0.0},
    {'a': -0.425, 'alpha': 0.0,        'd': 0.0,    'theta_offset': 0.0},
    {'a': -0.3922,'alpha': 0.0,        'd': 0.0,    'theta_offset': 0.0},
    {'a': 0.0,    'alpha': math.pi/2,  'd': 0.1333, 'theta_offset': 0.0},
    {'a': 0.0,    'alpha': -math.pi/2, 'd': 0.0997, 'theta_offset': 0.0},
    {'a': 0.0,    'alpha': 0.0,        'd': 0.0996, 'theta_offset': 0.0}
]
TCP_Z_OFFSET = 0.1565
H_FLANGE_TCP = np.array([
    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, TCP_Z_OFFSET], [0, 0, 0, 1]
], dtype=float)

def dh_matrix(a, alpha, d, theta):
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    return np.array([
        [cos_t, -sin_t*cos_a,  sin_t*sin_a, a*cos_t],
        [sin_t,  cos_t*cos_a, -cos_t*sin_a, a*sin_t],
        [    0,         sin_a,         cos_a,         d],
        [    0,             0,             0,         1]
    ])

def forward_kinematics(joint_angles_rad, dh_params=DH_PARAMS_UR5E, H_flange_tcp=H_FLANGE_TCP):
    if len(joint_angles_rad) != len(dh_params): return None, None
    try:
        transforms = [np.identity(4)]; T_prev = transforms[0]
        for i in range(len(dh_params)):
            p = dh_params[i]
            T_i_minus_1_to_i = dh_matrix(p['a'], p['alpha'], p['d'], joint_angles_rad[i] + p['theta_offset'])
            T_curr = T_prev @ T_i_minus_1_to_i
            transforms.append(T_curr); T_prev = T_curr
        T0_flange = transforms[-1]; T0_TCP = T0_flange @ H_flange_tcp
        joint_positions = [T[:3, 3] for T in transforms]
        joint_positions.append(T0_TCP[:3, 3])
        return joint_positions, T0_TCP
    except Exception: return None, None

# --- Interpolation Function (Keep as is) ---
def interpolate_trajectory(trajectory_rad, num_points_out):
    num_points_in, num_joints = trajectory_rad.shape
    if num_points_in < 2: return trajectory_rad
    diffs = np.diff(trajectory_rad, axis=0); diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
    distances = np.linalg.norm(diffs, axis=1); cumulative_dist = np.concatenate(([0], np.cumsum(distances)))
    if cumulative_dist[-1] < 1e-6: return np.tile(trajectory_rad[0], (num_points_out, 1))
    interp_times = np.linspace(0, cumulative_dist[-1], num_points_out)
    unwrapped_traj = np.unwrap(trajectory_rad, axis=0)
    interpolated_traj = np.zeros((num_points_out, num_joints))
    for j in range(num_joints): interpolated_traj[:, j] = np.interp(interp_times, cumulative_dist, unwrapped_traj[:, j])
    return interpolated_traj

# --- YAML Loading Helpers (Keep as is) ---
def load_yaml_file(filepath: Path) -> Optional[dict]:
    if not filepath.is_file(): print(f"ERROR: YAML file not found: {filepath}"); return None
    try:
        with open(filepath, 'r') as f: data = yaml.safe_load(f)
        return data if data is not None else {}
    except Exception as e: print(f"ERROR reading/parsing YAML {filepath}: {e}"); return None

def load_interface_transforms(filepath: Path) -> Optional[dict[str, np.ndarray]]:
    config_data = load_yaml_file(filepath);
    if config_data is None: return None
    transforms = {}; loaded_count = 0
    for iface_id, data in config_data.items():
        try:
            if isinstance(data, dict) and 'T_aruco_interface' in data:
                H_D_I = np.array(data['T_aruco_interface'], dtype=float)
                if H_D_I.shape == (4, 4): transforms[iface_id] = H_D_I; loaded_count += 1
        except Exception: pass
    print(f"Loaded {loaded_count} interface transforms from {filepath.name}.")
    return transforms if transforms else None

def load_environment_config(filepath: Path) -> Optional[np.ndarray]:
    if not SCIPY_AVAILABLE: return None
    config_data = load_yaml_file(filepath);
    if config_data is None: return None
    try:
        dev_data = config_data.get('aruco_device', config_data.get('aruco_device_pose'))
        if dev_data is None: raise KeyError("Cannot find device key")
        pose_data = dev_data.get('pose');
        if pose_data is None: raise KeyError("Missing 'pose' key")
        H_B_D = matrix_from_pose_dict(pose_data)
        print(f"Loaded device pose (H_B_D) from {filepath.name}.")
        return H_B_D
    except Exception as e: print(f"ERROR extracting device pose from {filepath.name}: {e}"); return None

# --- NEW: Function to load boundary statistics (needs utils.py) ---
# We need to load utils.py dynamically like config.py to access its functions
utils_module = None
def load_utils_dynamically(config_dir: Path):
    """Loads the utils.py module from the specified directory."""
    global utils_module
    utils_py_path = config_dir.resolve() / "utils.py"
    if not utils_py_path.is_file():
        print(f"ERROR: utils.py not found in '{config_dir.resolve()}'")
        return False
    try:
        spec = importlib.util.spec_from_file_location("utils", utils_py_path)
        if spec is None or spec.loader is None:
             raise ImportError(f"Could not create spec for utils.py at {utils_py_path}")
        utils_module = importlib.util.module_from_spec(spec)
        sys.modules['utils'] = utils_module # Make it available globally as 'utils'
        # We need to load config first so utils can import it
        if 'config' not in sys.modules:
            print("ERROR: config module must be loaded before utils module.")
            return False
        spec.loader.exec_module(utils_module)
        print(f"Successfully loaded utils.py from {utils_py_path}")
        return True
    except Exception as e:
        print(f"ERROR loading utils.py: {e}")
        traceback.print_exc()
        utils_module = None
        return False

# --- Visualization Functions ---

# Store plotted elements globally
robot_lines_anim = []
tcp_frame_quivers_anim = []
tcp_frame_labels_anim = []
ellipsoid_lines_anim = [] # To store ellipsoid wireframes

def plot_coordinate_frame_static(ax, pose_matrix, length=0.1, labels=['X', 'Y', 'Z'], color=None, linewidth=1, text_offset=1.15):
    """ Plots a static 3D coordinate frame. (Keep as is) """
    if pose_matrix is None: return
    try:
        origin = pose_matrix[:3, 3]; R_mat = pose_matrix[:3, :3]; colors = color if color else ['r', 'g', 'b']
        for i in range(3):
            axis_vector = R_mat[:, i]; current_color = colors[i] if isinstance(colors, list) else colors
            ax.quiver(origin[0], origin[1], origin[2], axis_vector[0], axis_vector[1], axis_vector[2], length=length, color=current_color, arrow_length_ratio=0.3, linewidth=linewidth, normalize=False)
            if labels and len(labels) == 3:
                ax.text(origin[0] + axis_vector[0] * length * text_offset, origin[1] + axis_vector[1] * length * text_offset, origin[2] + axis_vector[2] * length * text_offset, labels[i], color=current_color, fontsize=9, ha='center', va='center', zorder=10)
    except Exception as e: print(f"Warning: Failed to plot static coordinate frame: {e}")

def update_robot_plot_anim(ax, q_rad, frame_index, total_frames):
    """ Clears previous robot plot and draws the robot. (Keep as is) """
    global robot_lines_anim, tcp_frame_quivers_anim, tcp_frame_labels_anim
    for artist_list in [robot_lines_anim, tcp_frame_quivers_anim, tcp_frame_labels_anim]:
        for artist in artist_list:
            try: artist.remove()
            except ValueError: pass
        artist_list.clear()
    joint_positions, T_tcp = forward_kinematics(q_rad)
    if joint_positions is not None and T_tcp is not None:
        jp = np.array(joint_positions)
        if jp.ndim == 2 and jp.shape[0] >= 2 and jp.shape[1] == 3:
            line, = ax.plot(jp[:,0], jp[:,1], jp[:,2], marker='o', ms=4, ls='-', lw=2, color='blue', alpha=0.8)
            robot_lines_anim.append(line)
        tcp_origin = T_tcp[:3, 3]; tcp_R_mat = T_tcp[:3, :3]
        tcp_colors = ['r', 'g', 'b']; tcp_labels = ['Xt', 'Yt', 'Zt']
        tcp_frame_length = 0.1; text_offset = 1.15
        for i in range(3):
            axis = tcp_R_mat[:, i]
            qv = ax.quiver(tcp_origin[0],tcp_origin[1],tcp_origin[2], axis[0],axis[1],axis[2], length=tcp_frame_length, color=tcp_colors[i], arrow_length_ratio=0.3, lw=1.5, normalize=False)
            tcp_frame_quivers_anim.append(qv)
            txt = ax.text(tcp_origin[0]+axis[0]*tcp_frame_length*text_offset, tcp_origin[1]+axis[1]*tcp_frame_length*text_offset, tcp_origin[2]+axis[2]*tcp_frame_length*text_offset, tcp_labels[i], color=tcp_colors[i], fontsize=9, ha='center', va='center', zorder=10)
            tcp_frame_labels_anim.append(txt)
    else: print(f"Warning: FK failed for frame {frame_index}, cannot update robot plot.")
    ax.set_title(f"Robot Trajectory Animation (Frame {frame_index+1}/{total_frames})")
    plt.draw()

# --- NEW: Plotting function for waypoints ---
def plot_waypoints(ax, q_trajectory, label, color, marker='o', size=30):
    """Calculates FK for a C-space trajectory and plots TCP markers."""
    if q_trajectory is None or len(q_trajectory) == 0:
        print(f"Warning: No data provided for plotting waypoints: {label}")
        return
    plotted_label = False
    for i, q in enumerate(q_trajectory):
        _, T_tcp = forward_kinematics(q)
        if T_tcp is not None:
            pos = T_tcp[:3, 3]
            current_label = label if not plotted_label else ""
            ax.scatter(pos[0], pos[1], pos[2], color=color, marker=marker, s=size, label=current_label, alpha=0.7, depthshade=True)
            plotted_label = True
        # else: print(f"Warning: FK failed for waypoint {i} in {label}") # Can be verbose

# --- NEW: Plotting function for 3D covariance ellipsoid ---
def plot_position_covariance_ellipsoid(ax, mean_pos, cov_3x3, n_std=1.0, color='grey', alpha=0.1, resolution=20, label=""):
    """Plots a 3D ellipsoid representing positional covariance."""
    if not SCIPY_AVAILABLE: return
    if mean_pos is None or cov_3x3 is None or cov_3x3.shape != (3, 3): return

    try:
        # Ensure covariance is symmetric and positive semi-definite
        cov_3x3 = (cov_3x3 + cov_3x3.T) / 2
        eigenvalues, eigenvectors = scipy.linalg.eigh(cov_3x3)

        # Check for near-zero eigenvalues
        if np.any(eigenvalues <= 1e-9):
             print(f"Warning: Covariance matrix for ellipsoid '{label}' is near singular. Skipping plot.")
             return

        # Radii corresponding to n_std deviations
        radii = n_std * np.sqrt(eigenvalues)

        # Generate points on a unit sphere
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        # Transform points to ellipsoid coordinates
        points = np.stack((x, y, z), axis=-1) # Shape (res, res, 3)
        points = points * radii # Scale by radii
        points = np.dot(points, eigenvectors.T) # Rotate
        points += mean_pos # Translate

        # Plot the wireframe
        # Plot wireframe along the first dimension (changes u)
        for i in range(len(v)):
            line, = ax.plot(points[i, :, 0], points[i, :, 1], points[i, :, 2], color=color, alpha=alpha*1.5, linewidth=0.5)
            ellipsoid_lines_anim.append(line)
        # Plot wireframe along the second dimension (changes v)
        for i in range(len(u)):
            line, = ax.plot(points[:, i, 0], points[:, i, 1], points[:, i, 2], color=color, alpha=alpha*1.5, linewidth=0.5)
            ellipsoid_lines_anim.append(line)

    except np.linalg.LinAlgError as e:
        print(f"Warning: LinAlgError plotting ellipsoid '{label}': {e}")
    except Exception as e:
        print(f"Warning: Failed to plot ellipsoid '{label}': {e}")
        # traceback.print_exc() # Optional: for more detail

# --- Main Execution Logic ---
if __name__ == "__main__":
    if not SCIPY_AVAILABLE: sys.exit(1)

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Animate robot trajectory from .npy file with static frames.")
    parser.add_argument("trajectory_filepath", type=str, help="Path to the final planned .npy file.")
    parser.add_argument("initial_guess_filepath", type=str, nargs='?', default="initial_guess_trajectory.npy", help="Path to initial guess .npy file (default: initial_guess_trajectory.npy).")
    parser.add_argument("optimized_waypoints_filepath", type=str, nargs='?', default="optimized_waypoints.npy", help="Path to optimized waypoints .npy file (default: optimized_waypoints.npy).")
    parser.add_argument("config_dir", type=str, nargs='?', default=".", help="Path to config directory (default: .).")
    parser.add_argument("num_interpolated_points", type=int, nargs='?', default=1000, help="Total animation frames (default: 200). Min: 2.")

    try:
        args = parser.parse_args()
        final_traj_path = Path(args.trajectory_filepath)
        initial_guess_path = Path(args.initial_guess_filepath)
        optimized_waypoints_path = Path(args.optimized_waypoints_filepath)
        config_dir = Path(args.config_dir)
        num_interp_points = args.num_interpolated_points
        if num_interp_points < 2: num_interp_points = 2
    except SystemExit: sys.exit(1)
    except Exception as e: print(f"Error parsing arguments: {e}"); sys.exit(1)

    # --- Load config.py ---
    config_py_path = config_dir.resolve() / "config.py"
    if not config_py_path.is_file(): print(f"ERROR: config.py not found in '{config_dir.resolve()}'"); sys.exit(1)
    try:
        spec = importlib.util.spec_from_file_location("config", config_py_path)
        if spec is None or spec.loader is None: raise ImportError(f"Could not create spec for config.py at {config_py_path}")
        config_module = importlib.util.module_from_spec(spec); sys.modules['config'] = config_module
        spec.loader.exec_module(config_module); print(f"Successfully loaded config.py from {config_py_path}")
    except Exception as e: print(f"ERROR loading config.py: {e}"); traceback.print_exc(); sys.exit(1)

    # --- Load utils.py ---
    if not load_utils_dynamically(config_dir):
        print("FATAL: Could not load utils.py. Cannot proceed.")
        sys.exit(1)
    # Now utils_module should be loaded and accessible

    # --- Load Transforms needed for static frames ---
    print("\nLoading transforms for static frames...")
    H_B_D = None; H_B_I = None; H_I_B = None # Need H_I_B for boundary stats
    try:
        iface_tf_path_str = config_module.INTERFACE_TRANSFORMS_YAML_PATH
        env_cfg_path_str = config_module.ENVIRONMENT_YAML_PATH
        stats_path_str = config_module.PROBABILISTIC_STATS_YAML_PATH # Get stats path
        interface_id = config_module.INTERFACE_ID

        config_py_dir = config_py_path.parent
        iface_tf_path = Path(iface_tf_path_str); env_cfg_path = Path(env_cfg_path_str); stats_path = Path(stats_path_str)
        if not iface_tf_path.is_absolute(): iface_tf_path = (config_py_dir / iface_tf_path).resolve()
        if not env_cfg_path.is_absolute(): env_cfg_path = (config_py_dir / env_cfg_path).resolve()
        if not stats_path.is_absolute(): stats_path = (config_py_dir / stats_path).resolve() # Resolve stats path

        print(f"  Interface Transforms: {iface_tf_path}")
        print(f"  Environment Config: {env_cfg_path}")
        print(f"  Boundary Stats: {stats_path}") # Print stats path

        if not iface_tf_path.is_file(): raise FileNotFoundError(f"Interface transforms file not found: {iface_tf_path}")
        if not env_cfg_path.is_file(): raise FileNotFoundError(f"Environment config file not found: {env_cfg_path}")
        if not stats_path.is_file(): raise FileNotFoundError(f"Boundary stats file not found: {stats_path}") # Check stats file

        interface_transforms = load_interface_transforms(iface_tf_path)
        H_B_D = load_environment_config(env_cfg_path)

        if interface_transforms is not None and H_B_D is not None:
            H_D_I = interface_transforms.get(interface_id)
            if H_D_I is None: raise ValueError(f"Interface ID '{interface_id}' not found in transforms file.")
            H_B_I = H_B_D @ H_D_I
            try: H_I_B = np.linalg.inv(H_B_I) # Calculate inverse needed for stats
            except np.linalg.LinAlgError: raise ValueError("Cannot invert H_B_I")
            print("  Successfully calculated H_B_D, H_B_I, H_I_B.")
        else: raise ValueError("Failed to load interface transforms or environment config.")

        # --- Load Boundary Statistics ---
        print("Loading boundary statistics...")
        # Use the dynamically loaded utils module
        boundary_stats_list = utils_module.load_boundary_statistics(str(stats_path))
        if boundary_stats_list is None:
            print("Warning: Failed to load boundary statistics. Means/Ellipsoids will not be plotted.")
        else:
             print(f"Successfully loaded {len(boundary_stats_list)} boundary statistics.")

    except Exception as e:
        print(f"ERROR loading transforms or stats: {e}")
        traceback.print_exc()
        print("Proceeding without plotting some static elements.")
        H_B_D = H_B_I = H_I_B = None # Ensure transforms are None
        boundary_stats_list = None # Ensure stats are None

    # --- Load Trajectory Data ---
    print("\nLoading trajectory files...")
    try:
        print(f"  Final path: {final_traj_path}")
        original_trajectory_rad = np.load(final_traj_path)
        if original_trajectory_rad.ndim != 2 or original_trajectory_rad.shape[1] != 6: raise ValueError("Final trajectory must be Nx6")
        if len(original_trajectory_rad) < 2: raise ValueError("Final trajectory needs at least 2 waypoints")

        # Load optional trajectories
        initial_guess_q_traj = None
        if initial_guess_path.is_file():
            print(f"  Initial guess: {initial_guess_path}")
            initial_guess_q_traj = np.load(initial_guess_path)
            if initial_guess_q_traj.ndim != 2 or initial_guess_q_traj.shape[1] != 6:
                print(f"Warning: Invalid shape {initial_guess_q_traj.shape} for initial guess file. Ignoring.")
                initial_guess_q_traj = None
        else: print(f"  Initial guess file not found: {initial_guess_path}")

        optimized_q_waypoints = None
        if optimized_waypoints_path.is_file():
            print(f"  Optimized waypoints: {optimized_waypoints_path}")
            optimized_q_waypoints = np.load(optimized_waypoints_path)
            if optimized_q_waypoints.ndim != 2 or optimized_q_waypoints.shape[1] != 6:
                 print(f"Warning: Invalid shape {optimized_q_waypoints.shape} for optimized waypoints file. Ignoring.")
                 optimized_q_waypoints = None
        else: print(f"  Optimized waypoints file not found: {optimized_waypoints_path}")

    except Exception as e: print(f"ERROR loading trajectory file: {e}"); sys.exit(1)

    # --- Interpolate Final Trajectory for Animation ---
    print(f"Interpolating final trajectory to {num_interp_points} points...")
    try: anim_trajectory_rad = interpolate_trajectory(original_trajectory_rad, num_interp_points)
    except Exception as e: print(f"ERROR interpolating trajectory: {e}"); sys.exit(1)

    # --- Setup Animation Plot ---
    print("Setting up animation plot...")
    fig_anim = plt.figure("Robot Animation", figsize=(12, 9)) # Slightly larger figure
    ax_anim = fig_anim.add_axes([0.05, 0.15, 0.9, 0.8], projection='3d')
    ax_anim.set_xlabel("X base (m)"); ax_anim.set_ylabel("Y base (m)"); ax_anim.set_zlabel("Z base (m)")
    plot_limits = getattr(config_module, 'VIS_PLOT_LIMITS', ([-1, 1], [-1, 1], [-0.2, 1.5]))
    ax_anim.set_xlim(plot_limits[0]); ax_anim.set_ylim(plot_limits[1]); ax_anim.set_zlim(plot_limits[2])
    ax_anim.set_aspect('equal', adjustable='box'); ax_anim.view_init(elev=30., azim=-60)

    # --- Plot Static Elements ---
    print("Plotting static elements...")
    legend_handles_static = []
    # Base Frame
    plot_coordinate_frame_static(ax_anim, np.identity(4), length=0.15, labels=['Xb', 'Yb', 'Zb'], linewidth=2)
    legend_handles_static.append(Line2D([0], [0], color='black', lw=2, label='Base Frame'))
    # Device Frame
    if H_B_D is not None:
        plot_coordinate_frame_static(ax_anim, H_B_D, length=0.12, labels=['Xd', 'Yd', 'Zd'], color='cyan', linewidth=2)
        legend_handles_static.append(Line2D([0], [0], color='cyan', lw=2, label='Device Frame'))
    # Interface Frame
    if H_B_I is not None:
        plot_coordinate_frame_static(ax_anim, H_B_I, length=0.10, labels=['Xi', 'Yi', 'Zi'], color='magenta', linewidth=2)
        legend_handles_static.append(Line2D([0], [0], color='magenta', lw=2, label='Interface Frame'))

    # Plot Boundary Means and Ellipsoids
    if boundary_stats_list is not None and H_B_I is not None:
        print("Plotting boundary means and ellipsoids...")
        plotted_mean_label = False
        plotted_ellipsoid_label = False
        for i, stats in enumerate(boundary_stats_list):
            mean_pos_I = stats.get('mean_pos_I')
            cov_12x12_I = stats.get('covariance_12x12_I')

            if mean_pos_I is not None:
                # Transform mean position to Base frame for plotting
                mean_pos_B = (H_B_I @ np.append(mean_pos_I, 1))[:3]
                label_mean = "Boundary Mean" if not plotted_mean_label else ""
                ax_anim.scatter(mean_pos_B[0], mean_pos_B[1], mean_pos_B[2], color='orange', marker='x', s=50, label=label_mean)
                plotted_mean_label = True

            if cov_12x12_I is not None and mean_pos_B is not None:
                # Extract 3x3 positional covariance
                cov_pos_I = cov_12x12_I[:3, :3]
                # Covariance needs to be transformed to the Base frame
                # Rotation part of H_B_I
                R_B_I = H_B_I[:3, :3]
                cov_pos_B = R_B_I @ cov_pos_I @ R_B_I.T # Transform covariance: R * Cov_I * R^T
                label_ell = "Boundary Cov (Pos)" if not plotted_ellipsoid_label else ""
                plot_position_covariance_ellipsoid(ax_anim, mean_pos_B, cov_pos_B, n_std=1.0, color='orange', alpha=0.1, label=label_ell)
                plotted_ellipsoid_label = True # Only add label once

        if plotted_mean_label: legend_handles_static.append(Line2D([0], [0], marker='x', color='orange', linestyle='', label='Boundary Mean'))
        if plotted_ellipsoid_label: legend_handles_static.append(Line2D([0], [0], color='orange', lw=1, alpha=0.3, label='Boundary Cov (Pos)'))

    # Plot Initial Guess Waypoints
    if initial_guess_q_traj is not None:
        print("Plotting initial guess waypoints...")
        plot_waypoints(ax_anim, initial_guess_q_traj, "Initial Guess TCP", 'red', marker='^', size=35)
        legend_handles_static.append(Line2D([0], [0], marker='^', color='red', linestyle='', label='Initial Guess TCP'))

    # Plot Optimized Waypoints
    if optimized_q_waypoints is not None:
        print("Plotting optimized waypoints...")
        plot_waypoints(ax_anim, optimized_q_waypoints, "Optimized Waypoint TCP", 'green', marker='s', size=40)
        legend_handles_static.append(Line2D([0], [0], marker='s', color='green', linestyle='', label='Optimized Waypoint TCP'))

    # Add legend for all static elements
    if legend_handles_static:
        ax_anim.legend(handles=legend_handles_static, loc='upper right', fontsize='small')

    # --- Add Slider ---
    ax_slider_anim = fig_anim.add_axes([0.15, 0.05, 0.7, 0.03])
    total_frames_anim = len(anim_trajectory_rad)
    slider_anim = Slider(ax=ax_slider_anim, label='Frame', valmin=0, valmax=total_frames_anim - 1, valinit=0, valstep=1, valfmt='%d')

    # --- Slider Update Function ---
    def update_anim(val):
        frame_index = int(slider_anim.val); frame_index = max(0, min(frame_index, total_frames_anim - 1))
        q_current = anim_trajectory_rad[frame_index]
        update_robot_plot_anim(ax_anim, q_current, frame_index, total_frames_anim)

    slider_anim.on_changed(update_anim)

    # --- Initial Plot ---
    print("Plotting initial animation frame...")
    update_anim(0) # Draw robot at initial frame

    # --- Show Plot ---
    print("\nDisplaying animation window... Use slider to navigate.")
    plt.show()

    print("\nAnimation script finished.")
