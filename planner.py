# planner.py
import numpy as np
import time
import config
from utils import normalize_angles # Assuming exists
import math
import numpy.linalg # For determinant calculation

# --- Import Jacobian calculation ---
try:
    from ik_solver import calculate_jacobian
    IK_SOLVER_AVAILABLE = True
    print("Successfully imported calculate_jacobian from ik_solver.")
except ImportError:
    print("\n--- WARNING ---")
    print("Could not import calculate_jacobian from ik_solver.py.")
    print("Manipulability objective will be disabled.")
    print("---------------\n")
    IK_SOLVER_AVAILABLE = False
    def calculate_jacobian(q): return None # Dummy function

# --- OMPL Imports ---
try:
    from ompl import base as ob
    from ompl import geometric as og
    config.OMPL_AVAILABLE = True # Update config based on successful import
    print("Successfully imported OMPL.")
except ImportError:
    print("\n--- ERROR: OMPL Import Failed ---")
    config.OMPL_AVAILABLE = False
# (Add other OMPL import exception handling if needed)


# --- OMPL State Validity Checker ---
if config.OMPL_AVAILABLE:
    class SimpleUR5eStateValidityChecker(ob.StateValidityChecker):
        # ... (Keep your existing State Validity Checker - MAKE SURE IT'S CORRECT) ...
        def __init__(self, si): super().__init__(si); self.si_ = si
        def isValid(self, state):
            q = np.array([state[i] for i in range(self.si_.getStateDimension())])
            if np.any(q < config.JOINT_LIMITS_MIN) or np.any(q > config.JOINT_LIMITS_MAX): return False
            # Add collision check here when ready: if not self.check_collisions(q): return False
            return True
        # def check_collisions(self, q): return True # Placeholder

    # --- OMPL Optimization Objectives ---

    class PathLengthObjective(ob.PathLengthOptimizationObjective):
        """ Standard path length objective. """
        def __init__(self, si):
            super().__init__(si)
            self.description_ = "Path Length"

    class ManipulabilityObjective(ob.OptimizationObjective):
        """ Custom OMPL objective to penalize low manipulability (cost = 1/w). """
        def __init__(self, si, jacobian_func, epsilon=config.MANIPULABILITY_EPSILON, max_cost=config.MAX_MANIPULABILITY_COST):
            super().__init__(si)
            self.si_ = si
            self.jacobian_func = jacobian_func
            self.epsilon = epsilon
            self.max_cost = max_cost
            self.description_ = "Manipulability Cost (1/w)" # More specific description

        def get_manipulability_cost(self, q_np):
            """ Calculates the cost = 1 / (sqrt(det(JJ^T)) + eps). Higher cost = lower manipulability. """
            if not IK_SOLVER_AVAILABLE or self.jacobian_func is None: return 0.0 # No penalty if calc unavailable

            try:
                J = self.jacobian_func(q_np)
                if J is None or J.shape[0] != 6: return self.max_cost # Penalize invalid Jacobian
                JJT = J @ J.T
                det_JJT = np.linalg.det(JJT)
                if det_JJT < 1e-12: w = 0.0 # Treat near-zero as singular
                else: w = math.sqrt(det_JJT)
                cost = 1.0 / (w + self.epsilon)
                return min(cost, self.max_cost) # Clamp cost
            except Exception: # Catch LinAlgError or others
                return self.max_cost # Penalize errors heavily

        def stateCost(self, s):
            """ Cost associated with a single state (configuration). """
            q_np = np.array([s[i] for i in range(self.si_.getStateDimension())])
            manip_cost = self.get_manipulability_cost(q_np)
            return ob.Cost(manip_cost) # Return as OMPL Cost object

        def motionCost(self, s1, s2):
            """ Cost of motion between states (average state cost * distance). """
            # Approximation: Average state cost * distance
            cost1 = self.stateCost(s1).value()
            cost2 = self.stateCost(s2).value()
            avg_cost = (cost1 + cost2) / 2.0
            dist = self.si_.distance(s1, s2) # Use OMPL's C-space distance
            return ob.Cost(avg_cost * dist) # Return as OMPL Cost object

    # --- Add other objective classes here if needed (e.g., JointLimitObjective) ---
    # Example:
    # class JointLimitObjective(ob.OptimizationObjective):
    #     def __init__(self, si, epsilon=config.OBJECTIVE_JLIM_EPSILON):
    #         super().__init__(si)
    #         self.si_ = si
    #         self.epsilon = epsilon
    #         self.description_ = "Joint Limit Avoidance Cost"
    #     def calculate_joint_limit_cost(self, q_np): # Use the helper from utils/ik_solver
    #         # Assuming you have this helper function available:
    #         # from utils import calculate_joint_limit_cost
    #         # return calculate_joint_limit_cost(q_np, self.epsilon)
    #         # Placeholder implementation if helper not separate:
    #         cost = 0.0
    #         for i in range(config.NUM_JOINTS):
    #              cost += 1.0 / (abs(q_np[i] - config.JOINT_LIMITS_MIN[i]) + self.epsilon)
    #              cost += 1.0 / (abs(config.JOINT_LIMITS_MAX[i] - q_np[i]) + self.epsilon)
    #         return cost
    #     def stateCost(self, s):
    #         q_np = np.array([s[i] for i in range(self.si_.getStateDimension())])
    #         jlim_cost = self.calculate_joint_limit_cost(q_np)
    #         return ob.Cost(jlim_cost)
    #     def motionCost(self, s1, s2):
    #         cost1 = self.stateCost(s1).value(); cost2 = self.stateCost(s2).value()
    #         avg_cost = (cost1 + cost2) / 2.0; dist = self.si_.distance(s1, s2)
    #         return ob.Cost(avg_cost * dist)

    # --- OMPL Planner Class ---
    class OMPLPlanner:
        def __init__(self):
            """Initializes the OMPL planner environment."""
            self.si = None
            self.space = None
            self.objective = None # Store the objective created during setup
            if not config.OMPL_AVAILABLE:
                print("OMPL not available, planner cannot be initialized.")
                return
            self._setup_space_information() # Calls _get_optimization_objective

        def _setup_space_information(self):
            """Sets up the OMPL state space, validity checker, and stores the objective."""
            try:
                self.space = ob.RealVectorStateSpace(config.NUM_JOINTS)
                bounds = ob.RealVectorBounds(config.NUM_JOINTS)
                for i in range(config.NUM_JOINTS):
                    bounds.setLow(i, config.JOINT_LIMITS_MIN[i])
                    bounds.setHigh(i, config.JOINT_LIMITS_MAX[i])
                self.space.setBounds(bounds)

                self.si = ob.SpaceInformation(self.space)
                validity_checker = SimpleUR5eStateValidityChecker(self.si)
                self.si.setStateValidityChecker(validity_checker)

                # --- Create and STORE the optimization objective ONCE ---
                self.objective = self._get_optimization_objective()
                if self.objective:
                    # Print type/description for confirmation
                    desc = self.objective.getDescription() if hasattr(self.objective, 'getDescription') and self.objective.getDescription() else type(self.objective).__name__
                    print(f"Planner configured with objective: '{desc}'")
                else: print("Warning: No optimization objective created for planner.")

                self.si.setup() # Finalize SI setup
                print("OMPL SpaceInformation setup complete.")

            except Exception as e:
                print(f"Error setting up OMPL SpaceInformation: {e}")
                self.si = None; self.space = None; self.objective = None

        def _get_optimization_objective(self):
            """Creates the optimization objective based on config weights."""
            if self.si is None: return None

            active_objectives = [] # List to store (objective_instance, weight) tuples

            # 1. Path Length
            if config.WEIGHT_PATH_LENGTH > 0:
                obj = PathLengthObjective(self.si)
                active_objectives.append((obj, config.WEIGHT_PATH_LENGTH))
                print(f"  Adding Objective: {obj.getDescription()} (Weight: {config.WEIGHT_PATH_LENGTH})")

            # 2. Manipulability
            if config.WEIGHT_MANIPULABILITY > 0:
                if IK_SOLVER_AVAILABLE:
                    obj = ManipulabilityObjective(self.si, calculate_jacobian)
                    active_objectives.append((obj, config.WEIGHT_MANIPULABILITY))
                    print(f"  Adding Objective: {obj.getDescription()} (Weight: {config.WEIGHT_MANIPULABILITY})")
                else:
                    print("  Skipping Manipulability Objective: Jacobian function unavailable.")

            # 3. Joint Limits (Example - Uncomment and ensure class exists if using)
            # if hasattr(config, 'WEIGHT_JLIM_OBJECTIVE') and config.WEIGHT_JLIM_OBJECTIVE > 0:
            #    try:
            #       obj = JointLimitObjective(self.si) # Add epsilon if needed
            #       active_objectives.append((obj, config.WEIGHT_JLIM_OBJECTIVE))
            #       print(f"  Adding Objective: {obj.getDescription()} (Weight: {config.WEIGHT_JLIM_OBJECTIVE})")
            #    except NameError:
            #       print("  Skipping Joint Limit Objective: Class 'JointLimitObjective' not defined.")


            # --- Combine Objectives ---
            if len(active_objectives) == 0:
                print("Warning: No objectives active (all weights zero?). Defaulting to Path Length.")
                return PathLengthObjective(self.si) # Default if nothing else selected
            elif len(active_objectives) == 1:
                print("Using single objective.")
                return active_objectives[0][0] # Return the single objective instance
            else:
                print("Creating MultiOptimizationObjective...")
                multi_obj = ob.MultiOptimizationObjective(self.si)
                for obj, weight in active_objectives:
                    multi_obj.addObjective(obj, weight)
                return multi_obj

        def plan(self, q_start_np, q_goal_np, planner_type="RRTConnect", time_limit=None):
            """ Plans a path using the configured objective stored in self.objective. """
            if not config.OMPL_AVAILABLE or self.si is None or self.objective is None:
                print("Error: OMPL not available, SI not setup, or Objective not created.")
                return None, float('inf')

            current_planning_time = time_limit if time_limit is not None else config.PLANNING_TIME_LIMIT

            pdef = ob.ProblemDefinition(self.si) # Create Problem Definition for this query
            start_state = ob.State(self.space); goal_state = ob.State(self.space)
            for i in range(config.NUM_JOINTS):
                start_state[i] = q_start_np[i]
                goal_state[i] = q_goal_np[i]
            pdef.setStartAndGoalStates(start_state, goal_state)

            # --- Set the stored objective on THIS problem definition ---
            pdef.setOptimizationObjective(self.objective)

            # Choose planner (ensure type exists)
            try: planner_class = getattr(og, planner_type)
            except AttributeError: print(f"Error: Planner type '{planner_type}' not found."); return None, float('inf')
            planner = planner_class(self.si)

            planner.setProblemDefinition(pdef)
            planner.setup()

            # Solve
            pdef.clearSolutionPaths()
            # print(f"Starting OMPL planning ({planner_type})...") # Less verbose
            solved = planner.solve(current_planning_time)

            # Process Solution
            solution_path_np = None
            solution_cost = float('inf')

            if solved:
                # print("Planning successful!") # Less verbose
                if pdef.hasApproximateSolution(): print(f"  Warning: Solution is approximate. Goal distance: {pdef.getSolutionDifference():.4f}")

                path_geometric = pdef.getSolutionPath()
                # Simplify Path (optional)
                simplifier = og.PathSimplifier(self.si); simplifier.simplifyMax(path_geometric)
                # Interpolate Path (optional)
                if config.INTERPOLATE_PATH_POINTS > 0: path_geometric.interpolate(config.INTERPOLATE_PATH_POINTS)

                # *** Get cost using the objective used for planning ***
                active_objective = pdef.getOptimizationObjective() # Should be self.objective
                if path_geometric and active_objective:
                    try:
                        ompl_cost = path_geometric.cost(active_objective)
                        cost_val = ompl_cost.value()
                        if cost_val is not None and np.isfinite(cost_val):
                             solution_cost = cost_val
                             # obj_desc = active_objective.getDescription() if hasattr(active_objective, 'getDescription') and active_objective.getDescription() else type(active_objective).__name__
                             # print(f"  Path Cost (Objective: {obj_desc}): {solution_cost:.4f}") # Less verbose
                        else: # Fallback
                             solution_cost = path_geometric.length()
                             # print(f"  Warning: Objective cost non-finite. Path Cost (Length): {solution_cost:.4f}") # Less verbose
                    except Exception: # Fallback
                         solution_cost = path_geometric.length()
                         # print(f"  Warning: Failed to get obj cost. Path Cost (Length): {solution_cost:.4f}") # Less verbose
                elif path_geometric: # Fallback
                     solution_cost = path_geometric.length()
                     # print(f"  Path Cost (Length): {solution_cost:.4f}") # Less verbose

                # Extract path to numpy array
                solution_path_np = self._extract_path_numpy(path_geometric, verbose=False)
            # else: print("Planning failed.") # Less verbose

            # Clean up (optional, might help memory in long runs)
            # pdef.clearSolutionPaths()
            # planner.clear()

            return solution_path_np, solution_cost

        def _extract_path_numpy(self, path_geometric, verbose=True):
            """ Extracts numpy arrays from an OMPL geometric path. """
            # (Keep the implementation from previous response)
            if self.si is None: return None
            if path_geometric is None or path_geometric.getStateCount() == 0: return None
            num_states = path_geometric.getStateCount(); dim = self.si.getStateDimension()
            try:
                path_np = np.array([[path_geometric.getState(i)[j] for j in range(dim)] for i in range(num_states)])
                return path_np
            except Exception as e: print(f"Error during path extraction: {e}"); return None

# --- Final module print statement ---
print("planner.py loaded (with OMPL objective handling).")