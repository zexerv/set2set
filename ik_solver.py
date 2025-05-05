#!/usr/bin/env python3
import numpy as np
import math



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
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, TCP_Z_OFFSET],
    [0, 0, 0, 1]
], dtype=float)

INV_H_FLANGE_TCP = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, -TCP_Z_OFFSET],
    [0, 0, 0, 1]
], dtype=float)

d1 = DH_PARAMS_UR5E[0]['d']
a2 = DH_PARAMS_UR5E[1]['a']
a3 = DH_PARAMS_UR5E[2]['a']
d4 = DH_PARAMS_UR5E[3]['d']
d5 = DH_PARAMS_UR5E[4]['d']
d6 = DH_PARAMS_UR5E[5]['d']
len_a2 = abs(a2)
len_a3 = abs(a3)

tol_zero      = 1e-9
tol_singularity = 1e-7
tol_compare   = 1e-6
tol_geom      = 1e-6
def normalize_angles(angles):
    """
    Normalizes joint angles to the range [-pi, pi].

    Args:
        angles: A list or numpy array of joint angles in radians.

    Returns:
        A list of normalized joint angles in radians within [-pi, pi].
    """
    normalized = []
    for angle in angles:
        # Use math.fmod for floating point remainder
        norm_angle = math.fmod(angle + math.pi, 2.0 * math.pi)
        if norm_angle < 0.0:
            norm_angle += 2.0 * math.pi
        normalized.append(norm_angle - math.pi)
    return normalized
def dh_matrix(a, alpha, d, theta):
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    return np.array([
        [cos_t, -sin_t*cos_a,  sin_t*sin_a, a*cos_t],
        [sin_t,  cos_t*cos_a, -cos_t*sin_a, a*sin_t],
        [    0,        sin_a,        cos_a,       d],
        [    0,            0,            0,       1]
    ])


def forward_kinematics(joint_angles, dh_params=DH_PARAMS_UR5E, H_flange_tcp=H_FLANGE_TCP):
    if len(joint_angles) != len(dh_params):
        raise ValueError(f"Angle count mismatch: Expected {len(dh_params)}, got {len(joint_angles)}")
    transforms = [np.identity(4)]
    T_prev = transforms[0]
    for i in range(len(dh_params)):
        p = dh_params[i]
        T_i_minus_1_to_i = dh_matrix(p['a'], p['alpha'], p['d'], joint_angles[i] + p['theta_offset'])
        T_curr = T_prev @ T_i_minus_1_to_i
        transforms.append(T_curr)
        T_prev = T_curr
    T0_flange = transforms[-1]
    T0_TCP = T0_flange @ H_flange_tcp
    transforms.append(T0_TCP)
    points = [T[:3, 3] for T in transforms]
    return points, T0_TCP, T0_flange

def inverse_kinematics(T_desired_TCP, dh_params=DH_PARAMS_UR5E, inv_H_flange_tcp=INV_H_FLANGE_TCP):
    solutions = []
    T_flange = T_desired_TCP @ inv_H_flange_tcp
    P60 = T_flange[:3, 3]
    R06 = T_flange[:3, :3]
    pxd, pyd, pzd = P60
    r11d, r12d, r13d = R06[0, :]
    r21d, r22d, r23d = R06[1, :]
    r31d, r32d, r33d = R06[2, :]
    P50 = P60 - d6 * R06[:, 2]
    P50x, P50y, P50z = P50
    A = P50y
    B = P50x
    dist_sq_xy = B**2 + A**2
    if dist_sq_xy < d4**2 - tol_geom:
        return []
    if abs(B) < tol_singularity and abs(A) < tol_singularity:
         return []
    sqrt_arg_t1 = max(0, dist_sq_xy - d4**2)
    sqrt_val_t1 = math.sqrt(sqrt_arg_t1)
    term1_t1 = math.atan2(B, -A)
    term2_t1 = math.atan2(sqrt_val_t1, d4)
    theta1_sol = [term1_t1 + term2_t1, term1_t1 - term2_t1]
    for t1_idx, t1 in enumerate(theta1_sol):
        s1, c1 = math.sin(t1), math.cos(t1)
        M_val = s1 * r13d - c1 * r23d
        M = np.clip(M_val, -1.0, 1.0)
        D = c1 * r22d - s1 * r12d
        E = s1 * r11d - c1 * r21d
        sqrt_arg_ED = E**2 + D**2
        if abs(M**2 + sqrt_arg_ED - 1.0) > tol_compare:
            continue
        sqrt_val_ED = max(0, sqrt_arg_ED)
        sqrt_val_ED = math.sqrt(sqrt_val_ED)
        if abs(sqrt_val_ED) < tol_singularity and abs(M) < tol_singularity:
            continue
        t5_sol = [math.atan2(sqrt_val_ED, M), math.atan2(-sqrt_val_ED, M)]
        for t5_idx, t5 in enumerate(t5_sol):
            s5, c5 = math.sin(t5), math.cos(t5)
            t6 = 0.0
            is_singular_s5 = abs(s5) < tol_singularity
            is_singular_c5 = abs(c5) < tol_singularity
            if is_singular_s5 or is_singular_c5:
                t6 = 0.0
            else:
                if abs(D) < tol_singularity and abs(E) < tol_singularity:
                     continue
                if s5 > 0:
                    t6 = math.atan2(D, E)
                else:
                    t6 = math.atan2(-D, -E)
            s6, c6 = math.sin(t6), math.cos(t6)
            F = c5 * c6
            C_val = c1 * r11d + s1 * r21d
            atan_y_234 = r31d * F - s6 * C_val
            atan_x_234 = F * C_val + s6 * r31d
            if abs(atan_y_234) < tol_singularity and abs(atan_x_234) < tol_singularity:
                continue
            t234 = math.atan2(atan_y_234, atan_x_234)
            s234, c234 = math.sin(t234), math.cos(t234)
            KC = c1*pxd + s1*pyd - s234*d5 + c234*s5*d6
            KS = pzd - d1 + c234*d5 + s234*s5*d6
            dist_sq_13 = KC**2 + KS**2
            denom_t3 = 2 * len_a2 * len_a3
            if abs(denom_t3) < tol_zero:
                 continue
            cos_t3_arg = (dist_sq_13 - len_a2**2 - len_a3**2) / denom_t3
            if abs(cos_t3_arg) > 1.0 + tol_geom:
                continue
            cos_t3_arg = np.clip(cos_t3_arg, -1.0, 1.0)
            sqrt_arg_t3 = 1.0 - cos_t3_arg**2
            sqrt_val_t3 = math.sqrt(max(0, sqrt_arg_t3))
            t3_sol = [math.atan2(sqrt_val_t3, cos_t3_arg), math.atan2(-sqrt_val_t3, cos_t3_arg)]
            for t3_idx, t3 in enumerate(t3_sol):
                s3 = math.sin(t3)
                c3 = cos_t3_arg
                if abs(KS) < tol_singularity and abs(KC) < tol_singularity:
                    continue
                term1_t2 = math.atan2(KS, KC)
                term2_y_t2 = a3 * s3
                term2_x_t2 = a2 + a3 * c3
                if abs(term2_y_t2) < tol_singularity and abs(term2_x_t2) < tol_singularity:
                     continue
                term2_t2 = math.atan2(term2_y_t2, term2_x_t2)
                t2 = term1_t2 - term2_t2
                t4 = t234 - t2 - t3
                sol_angles_raw = [t1, t2, t3, t4, t5, t6]
                sol_angles_normalized = [(a + math.pi) % (2 * math.pi) - math.pi for a in sol_angles_raw]
                try:
                    _, _, T0_flange_check = forward_kinematics(sol_angles_normalized, dh_params, H_FLANGE_TCP)
                    if np.allclose(T0_flange_check, T_flange, atol=tol_compare):
                        solutions.append((sol_angles_normalized, t1_idx, t5_idx, t3_idx))
                except ValueError:
                    continue
    return solutions


# Assume DH_PARAMS_UR5E and dh_matrix are defined as in your provided code
# Assume forward_kinematics is defined as in your provided code

def calculate_jacobian(q, dh_params=DH_PARAMS_UR5E):
    """
    Calculates the 6x6 geometric Jacobian for the UR5e flange frame.

    Args:
        q (np.array or list): Joint configuration (6 angles in radians).
        dh_params (list, optional): The DH parameters for the robot.
                                     Defaults to DH_PARAMS_UR5E.

    Returns:
        np.ndarray: The 6x6 geometric Jacobian matrix [[Jv], [Jw]],
                    or None if FK fails or input is invalid.
    """
    n = len(dh_params)
    if len(q) != n:
        raise ValueError(f"Jacobian calculation expected {n} joint angles, got {len(q)}")

    # 1. Calculate all intermediate transformations using FK logic
    # We need T_0^0, T_0^1, ..., T_0^6 (flange frame)
    transforms_0_i = [np.identity(4)] # T_0^0
    T_prev = transforms_0_i[0]
    try:
        for i in range(n):
            p = dh_params[i]
            # Use the provided dh_matrix function
            T_i_minus_1_to_i = dh_matrix(p['a'], p['alpha'], p['d'], q[i] + p['theta_offset'])
            T_curr = T_prev @ T_i_minus_1_to_i
            transforms_0_i.append(T_curr)
            T_prev = T_curr
        T0_flange = transforms_0_i[-1] # This is T_0^6
    except Exception as e:
        print(f"Error during forward kinematics within Jacobian calculation: {e}")
        return None # Cannot calculate Jacobian if FK fails

    # 2. Extract necessary vectors
    p_n = T0_flange[:3, 3] # Position of the end-effector (flange) w.r.t base
    jacobian = np.zeros((6, n)) # Initialize 6xN Jacobian

    # 3. Calculate Jacobian columns iteratively
    for i in range(n):
        T_0_i = transforms_0_i[i] # Transformation T_0^{i-1} for joint i (since loop is 0 to n-1)

        # z_{i-1}: Z-axis of frame {i-1} expressed in base frame {0}
        # This is the 3rd column of the rotation matrix part of T_0^{i-1}
        z_i_minus_1 = T_0_i[:3, 2]

        # p_{i-1}: Position of the origin of frame {i-1} w.r.t base frame {0}
        p_i_minus_1 = T_0_i[:3, 3]

        # Calculate linear velocity component Jv_i = z_{i-1} x (p_n - p_{i-1})
        # (Since all joints are revolute)
        Jv_i = np.cross(z_i_minus_1, (p_n - p_i_minus_1))

        # Calculate angular velocity component Jw_i = z_{i-1}
        # (Since all joints are revolute)
        Jw_i = z_i_minus_1

        # Assign to the i-th column of the Jacobian (0-indexed)
        jacobian[:3, i] = Jv_i
        jacobian[3:, i] = Jw_i

    return jacobian

# Example Usage (requires the rest of your ik_solver code):
# if __name__ == '__main__':
#     q_test = np.radians([0, -90, 90, -90, -90, 0])
#     J = calculate_jacobian(q_test)
#     if J is not None:
#         print("Calculated Jacobian:")
#         print(np.round(J, 3))
#         # Example: Calculate manipulability
#         try:
#             JJT_det = np.linalg.det(J @ J.T)
#             if JJT_det < 1e-6:
#                 manip = 0.0
#             else:
#                 manip = np.sqrt(JJT_det)
#             print(f"\nManipulability Measure: {manip:.4f}")
#         except np.linalg.LinAlgError:
#             print("\nSingular configuration encountered (Jacobian determinant is zero or matrix is not square/invertible).")
