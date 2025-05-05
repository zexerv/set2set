import numpy as np
import math

# --- Configuration File Paths ---
# Path for Probabilistic Approach (Boundary stats: mean pose + 12x12 covariance)
PROBABILISTIC_STATS_YAML_PATH = "/home/kadi/Desktop/Thesis/demo_processor_new/results/cross_section_stats.yaml" # <<< CONFIRM THIS PATH is correct and contains 12D stats
# Other paths
INTERFACE_TRANSFORMS_YAML_PATH = "/home/kadi/Desktop/Thesis/demo_processor_new/scripts/setset/interface_transforms.yaml"
ENVIRONMENT_YAML_PATH = "/home/kadi/Desktop/Thesis/demo_processor_new/scripts/setset/environment_config.yaml"

# --- Interface ID ---
INTERFACE_ID = "button_292" # <<< SET YOUR DESIRED INTERFACE ID HERE

# --- Sliding Window Local Optimization Configuration ---
# This approach optimizes C-space configurations (q) over a sliding window.

# Size of the sliding window (number of boundary points considered simultaneously)
# Example: If size is 3, the window includes [q_current, q_boundary_i, q_boundary_i+1, q_boundary_i+2]
SLIDING_WINDOW_SIZE = 3 # <<< TUNE THIS (e.g., 3 or 4)

# Parameters for the local optimizer used within each window
# (Specific algorithm implementation will be in optimization.py)
# All the methods for scipy optimizers are: 
# 'L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', and 'trust-krylov' 
LOCAL_OPTIMIZER_TYPE =  'L-BFGS-B' #"L-BFG-S" # Options: 'SLSQP', 'CustomGD', etc. (Placeholder for now)
LOCAL_OPT_MAX_ITERATIONS = 100 # Max iterations per window optimization
LOCAL_OPT_TOLERANCE = 1e-4 # Convergence tolerance for the optimizer

# Cost function weights for the local optimizer
# Cost = w_rrt * Cost_RRT + w_pose * Cost_PoseDeviation
LOCAL_OPT_WEIGHT_RRT_COST = 1.0 # Weight for OMPL planning cost (path length + manipulability) between window points
LOCAL_OPT_WEIGHT_POSE_DIST = 0.00001 # Weight for Mahalanobis distance from boundary mean pose (12D) (Higher = stay closer to demo mean)
LOCAL_OPT_WEIGHT_SEGMENT_COST = 0.1 #
# Planning time limit used *inside* the local optimizer's cost evaluation (per segment within the window)
LOCAL_OPT_EVAL_PLANNING_TIME = 0.001 # Keep this relatively short for fast evaluation
# Add this line to your config.py
NUM_OPTIMIZATION_PASSES = 10 # Number of times to repeat the full sliding window optimization
# --- Robot Configuration ---
# Add this line to your config.py
VISUALIZE_ROTATION_VALIDATION = True # Set to True to enable rotation validation comparison plot
NUM_JOINTS = 6
JOINT_LIMITS_MIN = np.deg2rad([-360] * NUM_JOINTS)
JOINT_LIMITS_MAX = np.deg2rad([ 360] * NUM_JOINTS)
VISUALIZE_OPTIMIZATION_STEPS = False # Set to True to generate plots showing optimizer steps per window
# Add this line to your config.py, typically near other visualization flags
VISUALIZE_OPTIMIZATION_COMPONENT_HISTORY = True # Set to True to enable component history plots
VISUALIZE_COMPONENT_INDEX = 0 # 0=tx, 1=ty, 2=tz (Base Frame TCP position component to plot)
# --- Planner Configuration ---
OMPL_AVAILABLE = False # Set automatically in planner.py based on import success

# Planning time limit used BETWEEN the final optimized waypoints (per segment)
# This is for generating the *final* trajectory after the window optimization is complete.
FINAL_SEGMENT_PLANNING_TIME = 0.5 # <<< Potentially allow more time for the final path

# --- OMPL Cost/Objective Function Weights (Used by the planner during cost evaluation) ---
WEIGHT_PATH_LENGTH = 10.0
WEIGHT_MANIPULABILITY = 0.0 # Enable/disable based on testing
MANIPULABILITY_EPSILON = 1e-6 # Adjusted epsilon
MAX_MANIPULABILITY_COST = 1e8
USE_OMPL_IN_LOCAL_COST = False
# --- IK Solution Selection Criteria Weights (Used for INITIAL branch selection) ---
# Selects the best branch based on cost to reach the *mean* of the *first* boundary.
IK_SELECT_WEIGHT_MANIP_COST = 1.0 # Weight for manipulability cost
IK_SELECT_WEIGHT_JLIM_COST  = 5.0 # Weight for joint limit proximity cost
IK_SELECT_WEIGHT_DIST_COST  = 1.0 # Weight for C-space distance cost from hint (q0)
IK_SELECT_JLIM_EPSILON      = 1e-4

# --- Path Processing ---
INTERPOLATE_PATH_POINTS = 50 # Number of points to interpolate final path (0 = disable)

# --- Visualization Configuration ---
VIS_3D_SNAPSHOTS = 10
VIS_PLOT_LIMITS = ([-1.0, 1.0], [-1.0, 1.0], [-0.2, 1.5])
VIS_BASE_FRAME_SIZE = 0.15
VIS_TCP_FRAME_SIZE = 0.1

# --- Print loaded configuration summary ---
print(f"--- config.py Loaded (Sliding Window Optimization Approach) ---")
print(f"Probabilistic Stats Path: {PROBABILISTIC_STATS_YAML_PATH}")
print(f"Interface ID: {INTERFACE_ID}")
print(f"Sliding Window Size: {SLIDING_WINDOW_SIZE}")
print(f"Local Optimizer Type: {LOCAL_OPTIMIZER_TYPE} (MaxIter: {LOCAL_OPT_MAX_ITERATIONS}, Tol: {LOCAL_OPT_TOLERANCE})")
print(f"Local Opt Cost Weights: RRT={LOCAL_OPT_WEIGHT_RRT_COST}, PoseDist(12D)={LOCAL_OPT_WEIGHT_POSE_DIST}")
print(f"Local Opt Planning Time (Eval): {LOCAL_OPT_EVAL_PLANNING_TIME}s")
print(f"Final Segment Planning Time: {FINAL_SEGMENT_PLANNING_TIME}s")
print(f"OMPL Cost Weights: PathLen={WEIGHT_PATH_LENGTH}, Manip={WEIGHT_MANIPULABILITY}")
print(f"Initial IK Branch Selection Weights: Manip={IK_SELECT_WEIGHT_MANIP_COST}, JLim={IK_SELECT_WEIGHT_JLIM_COST}, Dist={IK_SELECT_WEIGHT_DIST_COST}")
print(f"-------------------------------------------------------------")
