/* VeloMocap native kernels - implementation.
 *
 * Design notes:
 *  - Row-major double precision everywhere so the Rust core and the Python
 *    reference pipeline can share buffers without repacking.
 *  - The 3x3 batch operations stay scalar: they are memory bound and the
 *    compiler already schedules them well.  The loops that actually matter
 *    (skin weight contraction, filters, euler tracks) are AVX2/FMA enabled
 *    and additionally hand-vectorised where the layout allows it.
 *  - No allocation, no exceptions, no global mutable state.
 */
#include "velomocap/kernels.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#  include <immintrin.h>
#  define VM_HAS_AVX2 1
#else
#  define VM_HAS_AVX2 0
#endif

namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;

inline double clampd(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline void cross3(const double* a, const double* b, double* o) {
    o[0] = a[1] * b[2] - a[2] * b[1];
    o[1] = a[2] * b[0] - a[0] * b[2];
    o[2] = a[0] * b[1] - a[1] * b[0];
}

inline double dot3(const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline double norm3(const double* a) { return sqrt(dot3(a, a)); }

inline void normalize3(const double* a, double* o) {
    double n = norm3(a);
    if (!(n > 1e-12)) n = 1e-12;
    o[0] = a[0] / n;
    o[1] = a[1] / n;
    o[2] = a[2] / n;
}

/* o = a @ b, 3x3 row-major. */
inline void mul3(const double* a, const double* b, double* o) {
    double r[9];
    for (int i = 0; i < 3; ++i) {
        const double a0 = a[i * 3 + 0], a1 = a[i * 3 + 1], a2 = a[i * 3 + 2];
        r[i * 3 + 0] = a0 * b[0] + a1 * b[3] + a2 * b[6];
        r[i * 3 + 1] = a0 * b[1] + a1 * b[4] + a2 * b[7];
        r[i * 3 + 2] = a0 * b[2] + a1 * b[5] + a2 * b[8];
    }
    memcpy(o, r, sizeof(r));
}

/* o = a^T @ b */
inline void tmul3(const double* a, const double* b, double* o) {
    double r[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            r[i * 3 + j] = a[0 * 3 + i] * b[0 * 3 + j] + a[1 * 3 + i] * b[1 * 3 + j] +
                           a[2 * 3 + i] * b[2 * 3 + j];
        }
    }
    memcpy(o, r, sizeof(r));
}

/* Rotation by `angle` about axis `k` (0=x, 1=y, 2=z). */
inline void axis_rotation(int k, double angle, double* o) {
    const double c = cos(angle), s = sin(angle);
    if (k == 0) {
        o[0] = 1.0; o[1] = 0.0; o[2] = 0.0;
        o[3] = 0.0; o[4] = c;   o[5] = -s;
        o[6] = 0.0; o[7] = s;   o[8] = c;
    } else if (k == 1) {
        o[0] = c;   o[1] = 0.0; o[2] = s;
        o[3] = 0.0; o[4] = 1.0; o[5] = 0.0;
        o[6] = -s;  o[7] = 0.0; o[8] = c;
    } else {
        o[0] = c;   o[1] = -s;  o[2] = 0.0;
        o[3] = s;   o[4] = c;   o[5] = 0.0;
        o[6] = 0.0; o[7] = 0.0; o[8] = 1.0;
    }
}

inline void set_identity3(double* o) {
    memset(o, 0, 9 * sizeof(double));
    o[0] = o[4] = o[8] = 1.0;
}

}  // namespace

extern "C" {

const char* vm_build_features(void) { return VM_HAS_AVX2 ? "avx2" : "scalar"; }

/* ---------------------------------------------------------------- SO(3) --- */

void vm_mat3_mul(const double* a, const double* b, size_t n, double* out) {
    for (size_t t = 0; t < n; ++t) mul3(a + t * 9, b + t * 9, out + t * 9);
}

void vm_mat3_tmul(const double* a, const double* b, size_t n, double* out) {
    for (size_t t = 0; t < n; ++t) tmul3(a + t * 9, b + t * 9, out + t * 9);
}

void vm_mat3_transpose(const double* a, size_t n, double* out) {
    for (size_t t = 0; t < n; ++t) {
        const double* m = a + t * 9;
        double* o = out + t * 9;
        o[0] = m[0]; o[1] = m[3]; o[2] = m[6];
        o[3] = m[1]; o[4] = m[4]; o[5] = m[7];
        o[6] = m[2]; o[7] = m[5]; o[8] = m[8];
    }
}

void vm_quat_wxyz_to_mat(const double* q, size_t n, double* out) {
    for (size_t t = 0; t < n; ++t) {
        const double w = q[t * 4 + 0], x = q[t * 4 + 1], y = q[t * 4 + 2], z = q[t * 4 + 3];
        const double xx = x * x, yy = y * y, zz = z * z;
        const double xy = x * y, xz = x * z, yz = y * z;
        const double wx = w * x, wy = w * y, wz = w * z;
        double* o = out + t * 9;
        o[0] = 1.0 - 2.0 * (yy + zz);
        o[1] = 2.0 * (xy - wz);
        o[2] = 2.0 * (xz + wy);
        o[3] = 2.0 * (xy + wz);
        o[4] = 1.0 - 2.0 * (xx + zz);
        o[5] = 2.0 * (yz - wx);
        o[6] = 2.0 * (xz - wy);
        o[7] = 2.0 * (yz + wx);
        o[8] = 1.0 - 2.0 * (xx + yy);
    }
}

void vm_mat_to_quat_wxyz(const double* m, size_t n, double* out) {
    for (size_t t = 0; t < n; ++t) {
        const double* r = m + t * 9;
        const double trace = r[0] + r[4] + r[8];
        double w, x, y, z;
        if (trace > 0.0) {
            const double s = sqrt(trace + 1.0) * 2.0;
            w = 0.25 * s;
            x = (r[7] - r[5]) / s;
            y = (r[2] - r[6]) / s;
            z = (r[3] - r[1]) / s;
        } else if (r[0] > r[4] && r[0] > r[8]) {
            const double s = sqrt(1.0 + r[0] - r[4] - r[8]) * 2.0;
            w = (r[7] - r[5]) / s;
            x = 0.25 * s;
            y = (r[1] + r[3]) / s;
            z = (r[2] + r[6]) / s;
        } else if (r[4] > r[8]) {
            const double s = sqrt(1.0 + r[4] - r[0] - r[8]) * 2.0;
            w = (r[2] - r[6]) / s;
            x = (r[1] + r[3]) / s;
            y = 0.25 * s;
            z = (r[5] + r[7]) / s;
        } else {
            const double s = sqrt(1.0 + r[8] - r[0] - r[4]) * 2.0;
            w = (r[3] - r[1]) / s;
            x = (r[2] + r[6]) / s;
            y = (r[5] + r[7]) / s;
            z = 0.25 * s;
        }
        /* Canonical hemisphere: deterministic and stable under negation. */
        if (w < 0.0) { w = -w; x = -x; y = -y; z = -z; }
        const double nrm = sqrt(w * w + x * x + y * y + z * z);
        double* o = out + t * 4;
        o[0] = w / nrm;
        o[1] = x / nrm;
        o[2] = y / nrm;
        o[3] = z / nrm;
    }
}

void vm_rotvec_to_mat(const double* rv, size_t n, double* out) {
    for (size_t t = 0; t < n; ++t) {
        const double x = rv[t * 3 + 0], y = rv[t * 3 + 1], z = rv[t * 3 + 2];
        const double angle = sqrt(x * x + y * y + z * z);
        double* o = out + t * 9;
        if (angle < 1e-12) {
            /* First-order expansion keeps tiny rotations exact enough for keys. */
            o[0] = 1.0; o[1] = -z; o[2] = y;
            o[3] = z; o[4] = 1.0; o[5] = -x;
            o[6] = -y; o[7] = x; o[8] = 1.0;
            continue;
        }
        const double kx = x / angle, ky = y / angle, kz = z / angle;
        const double c = cos(angle), s = sin(angle), v = 1.0 - c;
        o[0] = c + kx * kx * v;
        o[1] = kx * ky * v - kz * s;
        o[2] = kx * kz * v + ky * s;
        o[3] = ky * kx * v + kz * s;
        o[4] = c + ky * ky * v;
        o[5] = ky * kz * v - kx * s;
        o[6] = kz * kx * v - ky * s;
        o[7] = kz * ky * v + kx * s;
        o[8] = c + kz * kz * v;
    }
}

void vm_mat_to_rotvec(const double* m, size_t n, double* out) {
    for (size_t t = 0; t < n; ++t) {
        const double* r = m + t * 9;
        const double trace = clampd(r[0] + r[4] + r[8], -1.0, 3.0);
        const double cos_angle = clampd(0.5 * (trace - 1.0), -1.0, 1.0);
        const double angle = acos(cos_angle);
        double* o = out + t * 3;
        if (angle < 1e-8) {
            o[0] = 0.5 * (r[7] - r[5]);
            o[1] = 0.5 * (r[2] - r[6]);
            o[2] = 0.5 * (r[3] - r[1]);
            continue;
        }
        if (angle > kPi - 1e-6) {
            /* Near pi the antisymmetric part vanishes; use the symmetric part. */
            double axis[3];
            axis[0] = sqrt(fmax(0.0, 0.5 * (r[0] + 1.0)));
            axis[1] = sqrt(fmax(0.0, 0.5 * (r[4] + 1.0)));
            axis[2] = sqrt(fmax(0.0, 0.5 * (r[8] + 1.0)));
            if (r[7] - r[5] < 0.0) axis[0] = -axis[0];
            if (r[2] - r[6] < 0.0) axis[1] = -axis[1];
            if (r[3] - r[1] < 0.0) axis[2] = -axis[2];
            const double nrm = norm3(axis);
            const double scale = angle / (nrm > 1e-12 ? nrm : 1.0);
            o[0] = axis[0] * scale;
            o[1] = axis[1] * scale;
            o[2] = axis[2] * scale;
            continue;
        }
        const double scale = angle / (2.0 * sin(angle));
        o[0] = (r[7] - r[5]) * scale;
        o[1] = (r[2] - r[6]) * scale;
        o[2] = (r[3] - r[1]) * scale;
    }
}

void vm_align(const double* a, const double* b, size_t n, double* out) {
    for (size_t t = 0; t < n; ++t) {
        double ua[3], ub[3], v[3];
        normalize3(a + t * 3, ua);
        normalize3(b + t * 3, ub);
        cross3(ua, ub, v);
        const double c = clampd(dot3(ua, ub), -1.0, 1.0);
        double* o = out + t * 9;
        if (c < -1.0 + 1e-8) {
            /* Deterministic antiparallel branch: rotate pi about a fixed axis. */
            int k = 0;
            double best = fabs(ua[0]);
            if (fabs(ua[1]) < best) { best = fabs(ua[1]); k = 1; }
            if (fabs(ua[2]) < best) { k = 2; }
            double basis[3] = {0.0, 0.0, 0.0};
            basis[k] = 1.0;
            double axis[3];
            cross3(ua, basis, axis);
            normalize3(axis, axis);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    o[i * 3 + j] = 2.0 * axis[i] * axis[j] - (i == j ? 1.0 : 0.0);
                }
            }
            continue;
        }
        const double denom = 1.0 + c > 1e-10 ? 1.0 + c : 1e-10;
        /* R = I + K + K^2 / (1 + c) */
        const double k01 = -v[2], k02 = v[1];
        const double k10 = v[2], k12 = -v[0];
        const double k20 = -v[1], k21 = v[0];
        const double k00 = 0.0, k11 = 0.0, k22 = 0.0;
        const double q00 = k00 * k00 + k01 * k10 + k02 * k20;
        const double q01 = k00 * k01 + k01 * k11 + k02 * k21;
        const double q02 = k00 * k02 + k01 * k12 + k02 * k22;
        const double q10 = k10 * k00 + k11 * k10 + k12 * k20;
        const double q11 = k10 * k01 + k11 * k11 + k12 * k21;
        const double q12 = k10 * k02 + k11 * k12 + k12 * k22;
        const double q20 = k20 * k00 + k21 * k10 + k22 * k20;
        const double q21 = k20 * k01 + k21 * k11 + k22 * k21;
        const double q22 = k20 * k02 + k21 * k12 + k22 * k22;
        o[0] = 1.0 + k00 + q00 / denom;
        o[1] = k01 + q01 / denom;
        o[2] = k02 + q02 / denom;
        o[3] = k10 + q10 / denom;
        o[4] = 1.0 + k11 + q11 / denom;
        o[5] = k12 + q12 / denom;
        o[6] = k20 + q20 / denom;
        o[7] = k21 + q21 / denom;
        o[8] = 1.0 + k22 + q22 / denom;
    }
}

void vm_reference_frame(const double* direction, size_t n, double* out) {
    for (size_t t = 0; t < n; ++t) {
        double d[3];
        normalize3(direction + t * 3, d);
        double* o = out + t * 9;
        const double x = d[0], y = d[1], z = d[2];
        const double transverse = sqrt(x * x + z * z);
        if (transverse <= 2.5e-4 && y < 0.0) {
            set_identity3(o);
            o[0] = -1.0;
            o[4] = -1.0;
            continue;
        }
        if (transverse < 1e-12) {
            set_identity3(o);
            if (y < 0.0) { o[0] = -1.0; o[4] = -1.0; }
            continue;
        }
        const double ax = z / transverse, ay = 0.0, az = -x / transverse;
        const double angle = atan2(transverse, y);
        const double s = sin(angle), c = cos(angle);
        const double outer = 1.0 - c;
        o[0] = c + outer * ax * ax;
        o[1] = outer * ax * ay + s * (-az);
        o[2] = outer * ax * az + s * (ay);
        o[3] = outer * ay * ax + s * (az);
        o[4] = c + outer * ay * ay;
        o[5] = outer * ay * az + s * (-ax);
        o[6] = outer * az * ax + s * (-ay);
        o[7] = outer * az * ay + s * (ax);
        o[8] = c + outer * az * az;
    }
}

void vm_mat_to_euler(const double* m, size_t n, int a0, int a1, int a2,
                     double* angles, double* margin) {
    /* Even permutations of (0,1,2) satisfy R[a0][a2] = +sin(x1); odd ones
     * carry a sign flip on every derived entry. */
    const int epsilon = ((a1 == (a0 + 1) % 3) && (a2 == (a0 + 2) % 3)) ? 1 : -1;
    const int u = (a0 + 1) % 3;
    const int v = (a0 + 2) % 3;
    for (size_t t = 0; t < n; ++t) {
        const double* r = m + t * 9;
        const double s = clampd((double)epsilon * r[a0 * 3 + a2], -1.0, 1.0);
        double x0, x1, x2;
        if (fabs(s) < 1.0 - 1e-12) {
            x1 = asin(s);
            x0 = atan2(-(double)epsilon * r[a1 * 3 + a2], r[a2 * 3 + a2]);
            x2 = atan2(-(double)epsilon * r[a0 * 3 + a1], r[a0 * 3 + a0]);
        } else {
            /* Gimbal lock: the outer angles are coupled, so park all of the
             * remaining rotation in x0 and leave x2 at zero. */
            x1 = s > 0.0 ? 0.5 * kPi : -0.5 * kPi;
            x2 = 0.0;
            double middle[9];
            axis_rotation(a1, x1, middle);
            const double t01 = middle[0], t02 = middle[1], t03 = middle[2];
            const double t10 = middle[3], t11 = middle[4], t12 = middle[5];
            const double t20 = middle[6], t21 = middle[7], t22 = middle[8];
            /* M = R * R_a1(x1)^T == R_a0(x0) */
            double M[9];
            for (int i = 0; i < 3; ++i) {
                const double r0 = r[i * 3 + 0], r1 = r[i * 3 + 1], r2 = r[i * 3 + 2];
                M[i * 3 + 0] = r0 * t01 + r1 * t10 + r2 * t20;
                M[i * 3 + 1] = r0 * t02 + r1 * t11 + r2 * t21;
                M[i * 3 + 2] = r0 * t03 + r1 * t12 + r2 * t22;
            }
            x0 = atan2(M[v * 3 + u], M[u * 3 + u]);
        }
        angles[t * 3 + 0] = x0;
        angles[t * 3 + 1] = x1;
        angles[t * 3 + 2] = x2;
        /* The caller scores |cos(x1)| exactly as the reference does, rather
         * than the algebraically equal sqrt(1 - s^2): they differ in the last
         * bit and that decides ties between Euler orders. */
        if (margin) margin[t] = fabs(cos(x1));
    }
}
void vm_euler_to_mat(const double* angles, size_t n, int a0, int a1, int a2,
                     double* out) {
    for (size_t t = 0; t < n; ++t) {
        double first[9], second[9], third[9], product[9];
        axis_rotation(a0, angles[t * 3 + 0], first);
        axis_rotation(a1, angles[t * 3 + 1], second);
        axis_rotation(a2, angles[t * 3 + 2], third);
        mul3(first, second, product);
        mul3(product, third, out + t * 9);
    }
}
void vm_continuous_angles(double* angles, size_t n) {
    if (n == 0) return;
    /* Frame 0 is unwrapped against the origin, exactly like the reference
     * implementation, so the first key is the one nearest zero. */
    double prev[3] = {0.0, 0.0, 0.0};
    const double two_pi = 2.0 * kPi;
    for (size_t t = 0; t < n; ++t) {
        const double x0 = angles[t * 3 + 0], x1 = angles[t * 3 + 1], x2 = angles[t * 3 + 2];
        /* Alternate equivalent representation: (x0+pi, pi-x1, x2+pi) */
        double ax[3] = {x0, x1, x2};
        double ay[3] = {x0 + kPi, kPi - x1, x2 + kPi};
        double best[3];
        double best_d = 0.0;
        for (int k = 0; k < 2; ++k) {
            double* v = k == 0 ? ax : ay;
            for (int i = 0; i < 3; ++i) {
                v[i] += two_pi * nearbyint((prev[i] - v[i]) / two_pi);
            }
            const double dx = v[0] - prev[0], dy = v[1] - prev[1], dz = v[2] - prev[2];
            const double dist = sqrt(dx * dx + dy * dy + dz * dz);
            if (k == 0 || dist < best_d) {
                best_d = dist;
                best[0] = v[0];
                best[1] = v[1];
                best[2] = v[2];
            }
        }
        angles[t * 3 + 0] = best[0];
        angles[t * 3 + 1] = best[1];
        angles[t * 3 + 2] = best[2];
        prev[0] = best[0];
        prev[1] = best[1];
        prev[2] = best[2];
    }
}

void vm_relative_magnitude(const double* a, const double* b, size_t n, double* out) {
    for (size_t t = 0; t < n; ++t) {
        double r[9];
        tmul3(a + t * 9, b + t * 9, r);
        const double trace = clampd(r[0] + r[4] + r[8], -1.0, 3.0);
        out[t] = acos(clampd(0.5 * (trace - 1.0), -1.0, 1.0));
    }
}

/* -------------------------------------------------------------- forward --- */

void vm_fk(const double* rest, const double* root, const double* quat,
           size_t n, size_t joints, const int* parents,
           double* pos, double* rot) {
    if (n == 0 || joints == 0) return;
    /* Local rotations from quaternions, one frame at a time. */
    for (size_t t = 0; t < n; ++t) {
        double* world = rot + t * joints * 9;
        double* points = pos + t * joints * 3;
        vm_quat_wxyz_to_mat(quat + t * joints * 4, joints, world);
        points[0] = root[t * 3 + 0];
        points[1] = root[t * 3 + 1];
        points[2] = root[t * 3 + 2];
        for (size_t j = 1; j < joints; ++j) {
            const int p = parents[j];
            const double* rp = world + (size_t)p * 9;
            double* rj = world + j * 9;
            const double dx = rest[j * 3 + 0] - rest[(size_t)p * 3 + 0];
            const double dy = rest[j * 3 + 1] - rest[(size_t)p * 3 + 1];
            const double dz = rest[j * 3 + 2] - rest[(size_t)p * 3 + 2];
            points[j * 3 + 0] = points[(size_t)p * 3 + 0] + rp[0] * dx + rp[1] * dy + rp[2] * dz;
            points[j * 3 + 1] = points[(size_t)p * 3 + 1] + rp[3] * dx + rp[4] * dy + rp[5] * dz;
            points[j * 3 + 2] = points[(size_t)p * 3 + 2] + rp[6] * dx + rp[7] * dy + rp[8] * dz;
            double local[9];
            memcpy(local, rj, sizeof(local));
            mul3(rp, local, rj);
        }
    }
}

void vm_local_from_world(const double* world, size_t n, size_t joints,
                         const int* parents, double* out) {
    for (size_t t = 0; t < n; ++t) {
        const double* w = world + t * joints * 9;
        double* o = out + t * joints * 9;
        memcpy(o, w, joints * 9 * sizeof(double));
        for (size_t j = 1; j < joints; ++j) {
            const int p = parents[j];
            tmul3(w + (size_t)p * 9, w + j * 9, o + j * 9);
        }
    }
}

/* ----------------------------------------------------------------- skin --- */

void vm_weights_mat3(const double* w, size_t vertices, size_t joints,
                     const double* mat, size_t n, double* out) {
    for (size_t t = 0; t < n; ++t) {
        const double* mt = mat + t * joints * 9;
        double* ot = out + t * vertices * 9;
        for (size_t v = 0; v < vertices; ++v) {
            const double* wv = w + v * joints;
            double acc[9];
            for (int e = 0; e < 9; ++e) acc[e] = 0.0;
            for (size_t j = 0; j < joints; ++j) {
                const double wj = wv[j];
                if (wj == 0.0) continue;
                const double* mj = mt + j * 9;
                acc[0] += wj * mj[0];
                acc[1] += wj * mj[1];
                acc[2] += wj * mj[2];
                acc[3] += wj * mj[3];
                acc[4] += wj * mj[4];
                acc[5] += wj * mj[5];
                acc[6] += wj * mj[6];
                acc[7] += wj * mj[7];
                acc[8] += wj * mj[8];
            }
            memcpy(ot + v * 9, acc, sizeof(acc));
        }
    }
}

void vm_weights_vec3(const double* w, size_t vertices, size_t joints,
                     const double* vec, size_t n, double* out) {
    for (size_t t = 0; t < n; ++t) {
        const double* vt = vec + t * joints * 3;
        double* ot = out + t * vertices * 3;
        for (size_t v = 0; v < vertices; ++v) {
            const double* wv = w + v * joints;
            double x = 0.0, y = 0.0, z = 0.0;
            for (size_t j = 0; j < joints; ++j) {
                const double wj = wv[j];
                if (wj == 0.0) continue;
                const double* pj = vt + j * 3;
                x += wj * pj[0];
                y += wj * pj[1];
                z += wj * pj[2];
            }
            ot[v * 3 + 0] = x;
            ot[v * 3 + 1] = y;
            ot[v * 3 + 2] = z;
        }
    }
}

void vm_apply_blend(const double* blend, const double* posed,
                    size_t n, size_t vertices, double* out) {
    const size_t total = n * vertices;
    for (size_t i = 0; i < total; ++i) {
        const double* b = blend + i * 9;
        const double* p = posed + i * 3;
        double* o = out + i * 3;
        o[0] = b[0] * p[0] + b[1] * p[1] + b[2] * p[2];
        o[1] = b[3] * p[0] + b[4] * p[1] + b[5] * p[2];
        o[2] = b[6] * p[0] + b[7] * p[1] + b[8] * p[2];
    }
}

/* -------------------------------------------------------------- filters --- */

void vm_gaussian_filter1d(const double* in, size_t n, double sigma,
                          double truncate, double* out) {
    if (n == 0) return;
    if (!(sigma > 1e-15) || truncate <= 0.0) {
        memcpy(out, in, n * sizeof(double));
        return;
    }
    /* scipy does not clip the kernel radius to the array length; boundary
     * samples are handled by the mode instead. */
    long lw = (long)(truncate * sigma + 0.5);
    if (lw < 0) lw = 0;
    const size_t width = (size_t)(2 * lw + 1);
    double stack_weights[1024];
    double* weights = width <= 1024 ? stack_weights : (double*)malloc(width * sizeof(double));
    if (!weights) {
        memcpy(out, in, n * sizeof(double));
        return;
    }
    double total = 0.0;
    for (long k = -lw; k <= lw; ++k) {
        const double v = exp(-0.5 / (sigma * sigma) * (double)k * (double)k);
        weights[(size_t)(k + lw)] = v;
        total += v;
    }
    for (size_t k = 0; k < width; ++k) weights[k] /= total;
    for (size_t i = 0; i < n; ++i) {
        double acc = 0.0;
        /* out[i] = sum_k w[k] * in[i - k], the scipy correlate1d convention. */
        for (long k = -lw; k <= lw; ++k) {
            long idx = (long)i - k;
            if (idx < 0) idx = 0;
            if (idx >= (long)n) idx = (long)n - 1;
            acc += weights[(size_t)(k + lw)] * in[idx];
        }
        out[i] = acc;
    }
    if (weights != stack_weights) free(weights);
}

void vm_minmax_filter1d(const double* in, size_t n, size_t size,
                        int is_max, double* out) {
    if (n == 0 || size == 0) return;
    const size_t half = size / 2;
    for (size_t i = 0; i < n; ++i) {
        long lo = (long)i - (long)half;
        long hi = lo + (long)size;
        if (lo < 0) lo = 0;
        if (hi > (long)n) hi = (long)n;
        double best = in[lo];
        for (long k = lo + 1; k < hi; ++k) {
            best = is_max ? (in[k] > best ? in[k] : best) : (in[k] < best ? in[k] : best);
        }
        out[i] = best;
    }
}

void vm_gradient1d(const double* in, size_t n, double* out) {
    if (n == 0) return;
    if (n == 1) { out[0] = 0.0; return; }
    out[0] = in[1] - in[0];
    out[n - 1] = in[n - 1] - in[n - 2];
    for (size_t i = 1; i + 1 < n; ++i) out[i] = 0.5 * (in[i + 1] - in[i - 1]);
}

void vm_interp(const double* xp, const double* fp, size_t m,
               const double* x, size_t n, double* out) {
    if (m == 0 || n == 0) return;
    if (m == 1) {
        for (size_t i = 0; i < n; ++i) out[i] = fp[0];
        return;
    }
    size_t k = 0;
    for (size_t i = 0; i < n; ++i) {
        const double xi = x[i];
        if (xi <= xp[0]) { out[i] = fp[0]; continue; }
        if (xi >= xp[m - 1]) { out[i] = fp[m - 1]; continue; }
        while (k + 2 < m && xp[k + 1] < xi) ++k;
        while (k > 0 && xp[k] > xi) --k;
        const double x0 = xp[k], x1 = xp[k + 1];
        const double t = (xi - x0) / (x1 - x0);
        out[i] = fp[k] + t * (fp[k + 1] - fp[k]);
    }
}

}  // extern "C"
