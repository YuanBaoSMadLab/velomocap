/* VeloMocap native kernels.
 *
 * Numeric hot loops extracted from the RGB MoCap Python pipeline.  Every
 * function is a plain C ABI entry point so the Rust core can call it without
 * any C++ type leakage.  All buffers are caller-owned, row-major, and sized
 * exactly as documented; no allocation happens inside these kernels.
 */
#ifndef VELOMOCAP_KERNELS_H
#define VELOMOCAP_KERNELS_H

#include <stddef.h>

#if defined(_WIN32)
#  define VM_API __declspec(dllexport)
#else
#  define VM_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Build identity: "avx2" or "scalar". */
VM_API const char* vm_build_features(void);

/* ---------------------------------------------------------------- SO(3) --- */

/* out[t] = a[t] @ b[t]   (3x3 row-major, n matrices) */
VM_API void vm_mat3_mul(const double* a, const double* b, size_t n, double* out);
/* out[t] = a[t]^T @ b[t] */
VM_API void vm_mat3_tmul(const double* a, const double* b, size_t n, double* out);
/* out[t] = a[t]^T */
VM_API void vm_mat3_transpose(const double* a, size_t n, double* out);

/* Quaternion (w,x,y,z) -> rotation matrix. */
VM_API void vm_quat_wxyz_to_mat(const double* q, size_t n, double* out);
/* Rotation matrix -> quaternion (w,x,y,z), w >= 0, deterministic. */
VM_API void vm_mat_to_quat_wxyz(const double* m, size_t n, double* out);
/* Axis-angle vector -> rotation matrix. */
VM_API void vm_rotvec_to_mat(const double* rv, size_t n, double* out);
/* Rotation matrix -> axis-angle vector with |angle| <= pi. */
VM_API void vm_mat_to_rotvec(const double* m, size_t n, double* out);
/* Shortest-arc proper rotation taking unit(a) to unit(b). */
VM_API void vm_align(const double* a, const double* b, size_t n, double* out);
/* Zero-roll +Y bone frame for a bone direction (BVH reference basis). */
VM_API void vm_reference_frame(const double* direction, size_t n, double* out);

/* Intrinsic Tait-Bryan extraction for R = R_a0(x0) R_a1(x1) R_a2(x2).
 *
 * a0/a1/a2 are the rotation axis indices (0=x, 1=y, 2=z) in application
 * order, so any of the twelve orders is expressed by choosing them directly.
 * `angles` receives n*3 values in slot order and `margin` receives n values
 * |cos(x1)| (1 with no gimbal lock, 0 at lock). */
VM_API void vm_mat_to_euler(const double* m, size_t n, int a0, int a1, int a2,
                            double* angles, double* margin);
/* Inverse of vm_mat_to_euler: R = R_a0(x0) R_a1(x1) R_a2(x2). */
VM_API void vm_euler_to_mat(const double* angles, size_t n, int a0, int a1, int a2,
                            double* out);
/* Remove 2*pi jumps from an n*3 Euler track in place. */
VM_API void vm_continuous_angles(double* angles, size_t n);
/* Relative rotation angle (scipy Rotation.magnitude) of a @ b^T per sample. */
VM_API void vm_relative_magnitude(const double* a, const double* b, size_t n, double* out);

/* -------------------------------------------------------------- forward --- */

/* Forward kinematics from quaternions.
 * rest: joints*3, root: n*3, quat: n*joints*4 (w,x,y,z), parents: joints.
 * pos: n*joints*3 world joint positions, rot: n*joints*9 world rotations. */
VM_API void vm_fk(const double* rest, const double* root, const double* quat,
                  size_t n, size_t joints, const int* parents,
                  double* pos, double* rot);

/* Local rotations from world rotations: local[j] = world[parent]^T @ world[j]. */
VM_API void vm_local_from_world(const double* world, size_t n, size_t joints,
                                const int* parents, double* out);

/* ----------------------------------------------------------------- skin --- */

/* out[t,v] = sum_j w[v,j] * mat[t,j]     w: V*J, mat: n*J*9, out: n*V*9 */
VM_API void vm_weights_mat3(const double* w, size_t vertices, size_t joints,
                            const double* mat, size_t n, double* out);
/* out[t,v] = sum_j w[v,j] * vec[t,j]     w: V*J, vec: n*J*3, out: n*V*3 */
VM_API void vm_weights_vec3(const double* w, size_t vertices, size_t joints,
                            const double* vec, size_t n, double* out);
/* out[t,v] = blend[t,v] @ posed[t,v]     blend: n*V*9, posed: n*V*3 */
VM_API void vm_apply_blend(const double* blend, const double* posed,
                           size_t n, size_t vertices, double* out);

/* -------------------------------------------------------------- filters --- */

/* scipy.ndimage.gaussian_filter1d(..., mode='nearest') */
VM_API void vm_gaussian_filter1d(const double* in, size_t n, double sigma,
                                 double truncate, double* out);
/* scipy.ndimage.minimum_filter1d / maximum_filter1d (mode='nearest') */
VM_API void vm_minmax_filter1d(const double* in, size_t n, size_t size,
                               int is_max, double* out);
/* numpy.gradient(..., edge_order=1) */
VM_API void vm_gradient1d(const double* in, size_t n, double* out);
/* numpy.interp for a monotonically increasing integer-free x grid. */
VM_API void vm_interp(const double* xp, const double* fp, size_t m,
                      const double* x, size_t n, double* out);

#ifdef __cplusplus
}
#endif

#endif /* VELOMOCAP_KERNELS_H */
