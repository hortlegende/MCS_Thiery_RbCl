//here a (washed out) phase transition in the system at T=2K but not 4K is visible
//compile with gcc -O3 -fopenmp mc_simulation.c -o sim -lgsl -lgslcblas -lm in terminal
#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h> // Included OpenMP
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

// ============================================================
// 0) Parameters
// ============================================================

#define L 18
#define N (4 * L * L * L)

#define THETA_MAX 0.35
static double cosin;

static const double kB_meV = 8.617e-2;
static const double S_mag = 2.5;
static const double inv_S_mag = 1.0 / 2.5;

static const double mu0_SI = 4.0 * M_PI * 1e-7;
static const double kB_SI = 1.380649e-23;
static const double muB_SI = 9.2740100783e-24;
static const double muB_meV = 0.057883818;
static const double g_fac = 2.0023193;

static const double a = 13.82500;
static const double b = 9.91800;
static const double c = 7.10000;
static double A[3][3];
static double Vcell_A3;
static double V_SI;

static double basis_frac[4][3];

static const double J1 = -0.296;
static const double J2 = -0.342;
static const double J3 = -0.347;
static const double J4 = -0.310;
static const double J5 = -0.143;

// Dx and Dy are constant across sweeps
static double Dx = -0.0007;
static double Dy = 0.0;

// Read-only arrays (Safe for all threads)
static double position[N][3];
static int NNJ1[N];
static int NNJ2[N][2];
static int NNJ3[N];
static int NNJ4[N][2];
static int NNJ5[N][2];

// ============================================================
// Thread-Local State Struct
// ============================================================
// Bundles all mutable data so each thread has a private copy
typedef struct {
    double spin_arr[N][3];
    double B[3];
    double Dz; // Moved inside to prevent thread contention
    double T;  // Unique temperature for this thread's run
    gsl_rng* rng;
    long long total_moves;
    long long accepted_moves;
    double current_total_energy;
} ThreadState;

// ============================================================
// Utilities
// ============================================================

static double vec_dot(const double a[3], const double b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static double vec_norm(const double v[3]) {
    return sqrt(vec_dot(v, v));
}

static void vec_copy(double dst[3], const double src[3]) {
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
}

static void vec_cross(double out[3], const double a[3], const double b[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

// ============================================================
// 1) Lattice & basis
// ============================================================

static void init_lattice_constants(void) {
    A[0][0] = a; A[0][1] = 0.0; A[0][2] = 0.0;
    A[1][0] = 0.0; A[1][1] = b; A[1][2] = 0.0;
    A[2][0] = 0.0; A[2][1] = 0.0; A[2][2] = c;
    Vcell_A3 = a * b * c;
    V_SI = (double)(L * L * L) * Vcell_A3 * 1e-30;
}

static double frac_mod(double x) {
    double r = fmod(x, 1.0);
    if (r < 0.0) r += 1.0;
    return r;
}

static void init_basis(void) {
    const double x = 0.1175;
    const double z = 0.31024;
    double r1[3] = { frac_mod(x),      frac_mod(0.25), frac_mod(z) };
    double r2[3] = { frac_mod(-x + 0.5), frac_mod(0.75), frac_mod(z + 0.5) };
    double r3[3] = { frac_mod(-x),     frac_mod(0.75), frac_mod(-z) };
    double r4[3] = { frac_mod(x + 0.5),  frac_mod(0.25), frac_mod(-z + 0.5) };
    for (int i = 0; i < 3; i++) {
        basis_frac[0][i] = r1[i];
        basis_frac[1][i] = r2[i];
        basis_frac[2][i] = r3[i];
        basis_frac[3][i] = r4[i];
    }
}

static int coords_to_index(int ix, int iy, int iz, int s) {
    int cell_index = ix * L * L + iy * L + iz;
    return 4 * cell_index + s;
}

static void index_to_coords(int idx, int* ix, int* iy, int* iz, int* s) {
    int cell_index = idx / 4;
    *s = idx % 4;
    *iz = cell_index % L;
    *iy = (cell_index / L) % L;
    *ix = cell_index / (L * L);
}

static void init_positions(void) {
    int k = 0;
    for (int ix = 0; ix < L; ix++) {
        for (int iy = 0; iy < L; iy++) {
            for (int iz = 0; iz < L; iz++) {
                double shift[3] = { (double)ix, (double)iy, (double)iz };
                for (int s = 0; s < 4; s++) {
                    double rf[3];
                    rf[0] = basis_frac[s][0] + shift[0];
                    rf[1] = basis_frac[s][1] + shift[1];
                    rf[2] = basis_frac[s][2] + shift[2];
                    position[k][0] = A[0][0] * rf[0] + A[0][1] * rf[1] + A[0][2] * rf[2];
                    position[k][1] = A[1][0] * rf[0] + A[1][1] * rf[1] + A[1][2] * rf[2];
                    position[k][2] = A[2][0] * rf[0] + A[2][1] * rf[1] + A[2][2] * rf[2];
                    k++;
                }
            }
        }
    }
}

static void random_unit_vector(double v[3], gsl_rng* rng) {
    v[0] = gsl_ran_gaussian(rng, 1.0);
    v[1] = gsl_ran_gaussian(rng, 1.0);
    v[2] = gsl_ran_gaussian(rng, 1.0);
    double n = vec_norm(v);
    if (n == 0.0) { v[0] = 1.0; v[1] = 0.0; v[2] = 0.0; }
    else { v[0] /= n; v[1] /= n; v[2] /= n; }
}

static void set_random_spin(int k, ThreadState* st) {
    double u[3];
    random_unit_vector(u, st->rng);
    st->spin_arr[k][0] = S_mag * u[0];
    st->spin_arr[k][1] = S_mag * u[1];
    st->spin_arr[k][2] = S_mag * u[2];
}

static void gen_lattice(ThreadState* st) {
    for (int k = 0; k < N; k++) {
        set_random_spin(k, st);
    }
}

static void get_nn_J1(void) {
    for (int i = 0; i < N; i++) {
        int ix, iy, iz, s;
        index_to_coords(i, &ix, &iy, &iz, &s);
        if (s == 1) { NNJ1[i] = coords_to_index(ix, iy, iz, 2); }
        else if (s == 2) { NNJ1[i] = coords_to_index(ix, iy, iz, 1); }
        else if (s == 0) { NNJ1[i] = coords_to_index((ix - 1 + L) % L, iy, iz, 3); }
        else { NNJ1[i] = coords_to_index((ix + 1) % L, iy, iz, 0); }
    }
}

static void get_nn_J2(void) {
    for (int i = 0; i < N; i++) {
        int ix, iy, iz, s;
        index_to_coords(i, &ix, &iy, &iz, &s);
        int partner, iz2;
        if (s == 0) { partner = 1; iz2 = (iz + 1) % L; }
        else if (s == 1) { partner = 0; iz2 = (iz - 1 + L) % L; }
        else if (s == 2) { partner = 3; iz2 = (iz + 1) % L; }
        else { partner = 2; iz2 = (iz - 1 + L) % L; }
        NNJ2[i][0] = coords_to_index(ix, iy, iz, partner);
        NNJ2[i][1] = coords_to_index(ix, iy, iz2, partner);
    }
}

static void get_nn_J3(void) {
    for (int i = 0; i < N; i++) {
        int ix, iy, iz, s;
        index_to_coords(i, &ix, &iy, &iz, &s);
        if (s == 2) { NNJ3[i] = coords_to_index(ix, iy, (iz + 1) % L, 1); }
        else if (s == 1) { NNJ3[i] = coords_to_index(ix, iy, (iz - 1 + L) % L, 2); }
        else if (s == 0) { NNJ3[i] = coords_to_index((ix - 1 + L) % L, iy, (iz + 1) % L, 3); }
        else { NNJ3[i] = coords_to_index((ix + 1) % L, iy, (iz - 1 + L) % L, 0); }
    }
}

static void get_nn_J4(void) {
    for (int i = 0; i < N; i++) {
        int ix, iy, iz, s;
        index_to_coords(i, &ix, &iy, &iz, &s);
        NNJ4[i][0] = coords_to_index(ix, iy, (iz - 1 + L) % L, s);
        NNJ4[i][1] = coords_to_index(ix, iy, (iz + 1) % L, s);
    }
}

static void get_nn_J5(void) {
    for (int i = 0; i < N; i++) {
        int ix, iy, iz, s;
        index_to_coords(i, &ix, &iy, &iz, &s);
        int partner = (s + 2) % 4;
        NNJ5[i][0] = coords_to_index(ix, iy, iz, partner);
        NNJ5[i][1] = coords_to_index((ix - 1 + L) % L, iy, iz, partner);
    }
}

static void build_neighbors(void) {
    get_nn_J1(); get_nn_J2(); get_nn_J3(); get_nn_J4(); get_nn_J5();
}

static double total_energy(ThreadState* st) {
    double E_exchange = 0.0;
    double E_single = 0.0;
    for (int k = 0; k < N; k++) {
        int j;
        j = NNJ1[k];    E_exchange += -J1 * vec_dot(st->spin_arr[k], st->spin_arr[j]);
        j = NNJ2[k][0]; E_exchange += -J2 * vec_dot(st->spin_arr[k], st->spin_arr[j]);
        j = NNJ2[k][1]; E_exchange += -J2 * vec_dot(st->spin_arr[k], st->spin_arr[j]);
        j = NNJ3[k];    E_exchange += -J3 * vec_dot(st->spin_arr[k], st->spin_arr[j]);
        j = NNJ4[k][0]; E_exchange += -J4 * vec_dot(st->spin_arr[k], st->spin_arr[j]);
        j = NNJ4[k][1]; E_exchange += -J4 * vec_dot(st->spin_arr[k], st->spin_arr[j]);
        j = NNJ5[k][0]; E_exchange += -J5 * vec_dot(st->spin_arr[k], st->spin_arr[j]);
        j = NNJ5[k][1]; E_exchange += -J5 * vec_dot(st->spin_arr[k], st->spin_arr[j]);

        double sx = st->spin_arr[k][0], sy = st->spin_arr[k][1], sz = st->spin_arr[k][2];
        // Swapped to st->Dz
        E_single += +(Dx * sx * sx + Dy * sy * sy + st->Dz * sz * sz)
            - muB_meV * g_fac * (st->B[0] * sx + st->B[1] * sy + st->B[2] * sz);
    }
    return (0.5 * E_exchange) + E_single;
}

static void propose_cone_spin(const double current_spin[3], double out[3], gsl_rng* rng) {
    double v1[3];
    if (fabs(current_spin[0]) < fabs(current_spin[1])) {
        v1[0] = 1.0; v1[1] = 0.0; v1[2] = 0.0;
    }
    else {
        v1[0] = 0.0; v1[1] = 1.0; v1[2] = 0.0;
    }
    double e1[3], e2[3];
    vec_cross(e1, current_spin, v1);
    double n1 = vec_norm(e1);
    e1[0] /= n1; e1[1] /= n1; e1[2] /= n1;

    vec_cross(e2, current_spin, e1);
    double n2 = vec_norm(e2);
    e2[0] /= n2; e2[1] /= n2; e2[2] /= n2;

    double phi = gsl_rng_uniform(rng) * 2.0 * M_PI;
    double cos_theta = 1.0 - gsl_rng_uniform(rng) * (1.0 - cosin);
    double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    double sin_phi = sin(phi);
    double cos_phi = cos(phi);

    double u_curr[3] = {
        current_spin[0] * inv_S_mag,
        current_spin[1] * inv_S_mag,
        current_spin[2] * inv_S_mag
    };

    out[0] = cos_theta * u_curr[0] + sin_theta * (cos_phi * e1[0] + sin_phi * e2[0]);
    out[1] = cos_theta * u_curr[1] + sin_theta * (cos_phi * e1[1] + sin_phi * e2[1]);
    out[2] = cos_theta * u_curr[2] + sin_theta * (cos_phi * e1[2] + sin_phi * e2[2]);

    out[0] *= S_mag; out[1] *= S_mag; out[2] *= S_mag;
}

static double local_field_energy(int k, const double h[3], double spin[3], ThreadState* st) {
    double sx = spin[0], sy = spin[1], sz = spin[2];
    // Swapped to st->Dz
    return vec_dot(h, spin)
        + (Dx * sx * sx + Dy * sy * sy + st->Dz * sz * sz)
        - muB_meV * g_fac * (st->B[0] * sx + st->B[1] * sy + st->B[2] * sz);
}

static void metropolis_move(double beta, ThreadState* st) {
    int k = (int)(gsl_rng_uniform(st->rng) * N);
    if (k == N) k = N - 1;

    double h[3] = { 0,0,0 };
    int j;
    j = NNJ1[k];    h[0] += -J1 * st->spin_arr[j][0]; h[1] += -J1 * st->spin_arr[j][1]; h[2] += -J1 * st->spin_arr[j][2];
    j = NNJ2[k][0]; h[0] += -J2 * st->spin_arr[j][0]; h[1] += -J2 * st->spin_arr[j][1]; h[2] += -J2 * st->spin_arr[j][2];
    j = NNJ2[k][1]; h[0] += -J2 * st->spin_arr[j][0]; h[1] += -J2 * st->spin_arr[j][1]; h[2] += -J2 * st->spin_arr[j][2];
    j = NNJ3[k];    h[0] += -J3 * st->spin_arr[j][0]; h[1] += -J3 * st->spin_arr[j][1]; h[2] += -J3 * st->spin_arr[j][2];
    j = NNJ4[k][0]; h[0] += -J4 * st->spin_arr[j][0]; h[1] += -J4 * st->spin_arr[j][1]; h[2] += -J4 * st->spin_arr[j][2];
    j = NNJ4[k][1]; h[0] += -J4 * st->spin_arr[j][0]; h[1] += -J4 * st->spin_arr[j][1]; h[2] += -J4 * st->spin_arr[j][2];
    j = NNJ5[k][0]; h[0] += -J5 * st->spin_arr[j][0]; h[1] += -J5 * st->spin_arr[j][1]; h[2] += -J5 * st->spin_arr[j][2];
    j = NNJ5[k][1]; h[0] += -J5 * st->spin_arr[j][0]; h[1] += -J5 * st->spin_arr[j][1]; h[2] += -J5 * st->spin_arr[j][2];

    double old_spin[3]; vec_copy(old_spin, st->spin_arr[k]);
    double E_old = local_field_energy(k, h, old_spin, st);

    double new_spin[3];
    if (gsl_rng_uniform(st->rng) < 0.25) {
        double u[3];
        random_unit_vector(u, st->rng);
        new_spin[0] = S_mag * u[0];
        new_spin[1] = S_mag * u[1];
        new_spin[2] = S_mag * u[2];
    }
    else {
        propose_cone_spin(old_spin, new_spin, st->rng);
    }

    double E_new = local_field_energy(k, h, new_spin, st);
    double dE = E_new - E_old;
    st->total_moves++;

    if (dE > 0.0 && gsl_rng_uniform(st->rng) >= exp(-beta * dE)) {
        // rejected
    }
    else {
        vec_copy(st->spin_arr[k], new_spin);
        st->accepted_moves++;
        st->current_total_energy += dE;
    }
}

static void mcs_sweep(double beta, ThreadState* st) {
    for (int i = 0; i < N; i++) metropolis_move(beta, st);
}

static void magnetization_total(double* Mx, double* My, double* Mz, ThreadState* st) {
    double mx = 0.0, my = 0.0, mz = 0.0;
    for (int i = 0; i < N; i++) {
        mx += st->spin_arr[i][0]; my += st->spin_arr[i][1]; mz += st->spin_arr[i][2];
    }
    *Mx = mx; *My = my; *Mz = mz;
}

static void save_spin_snapshot(double current_T, double current_Dz, double current_B, int step, ThreadState* st) {
    char snap_filename[256];
    // Appended T, Dz, and step logic so filenames never overwrite.
    sprintf(snap_filename, "spins_T_%.1f_Dz_%.5f_B_%.3f_step_%d.xyz", current_T, current_Dz, current_B, step);

    FILE* fsnap = fopen(snap_filename, "w");
    if (!fsnap) return;

    fprintf(fsnap, "# Spin Snapshot: L=%d, T=%.1f, Dz=%.5f, B=%.3f T, Step=%d\n", L, current_T, current_Dz, current_B, step);
    fprintf(fsnap, "# x y z Sx Sy Sz\n");

    for (int i = 0; i < N; i++) {
        fprintf(fsnap, "% .4f % .4f % .4f % .6f % .6f % .6f\n",
            position[i][0], position[i][1], position[i][2],
            st->spin_arr[i][0], st->spin_arr[i][1], st->spin_arr[i][2]);
    }
    fclose(fsnap);
}

// ============================================================
// Hysteresis Task Run
// ============================================================
static void run_hysteresis_task(double current_T, double current_Dz) {
    char filename[128];
    sprintf(filename, "hysteresis_T_%.1f_Dz_%.5f.dat", current_T, current_Dz);

    FILE* fout = fopen(filename, "w");
    if (!fout) return;

    fprintf(fout, "# T=%.1f Dx=%.4f Dy=%.4f Dz=%.5f L=%d\n", current_T, Dx, Dy, current_Dz, L);
    fprintf(fout, "# B(T) m_x chi_x acc_ratio E/N(meV) Cv/N m_y m_z chi_y chi_z\n");

    ThreadState st;
    st.T = current_T;
    st.Dz = current_Dz;
    st.rng = gsl_rng_alloc(gsl_rng_taus2);

    // Seed generation taking parameters into account
    unsigned long seed = 12345UL + omp_get_thread_num() * 1000 + (unsigned long)(current_Dz * 1000000) + (unsigned long)(current_T * 10);
    gsl_rng_set(st.rng, seed);

    // Create Field Path (Low -> High -> Low)
    double B_start = 1.1;
    double B_peak = 1.7;
    double dB = 0.015;

    int steps_up = (int)(round((B_peak - B_start) / dB));
    int total_steps = 2 * steps_up + 1;
    double* B_path = (double*)malloc(total_steps * sizeof(double));

    // Populate Path
    for (int i = 0; i <= steps_up; i++) {
        B_path[i] = B_start + i * dB;
    }
    for (int i = 0; i < steps_up; i++) {
        B_path[steps_up + 1 + i] = B_peak - (i + 1) * dB;
    }

    double B_dir[3] = { 1.0, 0.0, 0.0 };

    // 1. Initialize lattice ONCE per hysteresis loop
    gen_lattice(&st);

    // 2. Initial heavy thermalization at the starting field
    st.B[0] = B_path[0] * B_dir[0];
    st.B[1] = B_path[0] * B_dir[1];
    st.B[2] = B_path[0] * B_dir[2];

    double beta = 1.0 / (kB_meV * st.T);
    int initial_eq = 11000;
    for (int i = 0; i < initial_eq; i++) mcs_sweep(beta, &st);

    // 3. Step through the generated path sequentially
    for (int step = 0; step < total_steps; step++) {
        double B_abs = B_path[step];
        st.B[0] = B_abs * B_dir[0];
        st.B[1] = B_abs * B_dir[1];
        st.B[2] = B_abs * B_dir[2];

        // Short thermalization as spins start off close to equilibrium 
        int n_eq = 3800;
        for (int i = 0; i < n_eq; i++) mcs_sweep(beta, &st);

        st.total_moves = 0;
        st.accepted_moves = 0;
        st.current_total_energy = total_energy(&st);

        int n_meas = 1100;
        int sample_every = 110;

        double E_sum = 0.0, E2_sum = 0.0;
        double Mx_sum = 0.0, Mx2_sum = 0.0;
        double My_sum = 0.0, My2_sum = 0.0;
        double Mz_sum = 0.0, Mz2_sum = 0.0;

        for (int m = 0; m < n_meas; m++) {
            for (int s = 0; s < sample_every; s++) mcs_sweep(beta, &st);

            double E = st.current_total_energy;
            E_sum += E; E2_sum += E * E;

            double Mx, My, Mz;
            magnetization_total(&Mx, &My, &Mz, &st);
            Mx_sum += Mx; Mx2_sum += Mx * Mx;
            My_sum += My; My2_sum += My * My;
            Mz_sum += Mz; Mz2_sum += Mz * Mz;
        }

        // Output Math
        double Mx_mean = Mx_sum / n_meas;
        double My_mean = My_sum / n_meas;
        double Mz_mean = Mz_sum / n_meas;
        double E_mean = E_sum / n_meas;

        double Mx2_mean = Mx2_sum / n_meas;
        double My2_mean = My2_sum / n_meas;
        double Mz2_mean = Mz2_sum / n_meas;
        double E2_mean = E2_sum / n_meas;

        double Cv_total = kB_meV * beta * beta * (E2_mean - E_mean * E_mean);
        double Cv_per_ion = Cv_total / (double)N;

        double chi_x = beta * (Mx2_mean - Mx_mean * Mx_mean);
        double chi_y = beta * (My2_mean - My_mean * My_mean);
        double chi_z = beta * (Mz2_mean - Mz_mean * Mz_mean);
        double acc_rate = (double)st.accepted_moves / st.total_moves;

        // Snapshots (Disabled to prevent excessive outputs unless B is exactly 1.0 or 2.0)
        if (fabs(B_abs - 1.0) < dB || fabs(B_abs - 2.0) < dB) {
            save_spin_snapshot(current_T, current_Dz, B_abs, step, &st);
        }

        // Live write-out keeps data safe in case of crashes
        fprintf(fout, " % 6.3f % .8e % .8e % .8e % .8e % .8e % .8e % .8e % .8e % .8e \n",
            B_abs, Mx_mean, chi_x,
            acc_rate * 100, E_mean / (double)N,
            Cv_per_ion, My_mean, Mz_mean,
            chi_y, chi_z);
        fflush(fout);
    }

    free(B_path);
    gsl_rng_free(st.rng);
    fclose(fout);
}

// ============================================================
// 12) Main Execution
// ============================================================

int main(void) {
    init_lattice_constants();
    init_basis();
    init_positions();
    build_neighbors();

    cosin = cos(THETA_MAX);

    // Setup T & Dz parameters
    double T_values[] = {2.0};
    int num_T = sizeof(T_values) / sizeof(T_values[0]);

    double Dz_start = -0.00070;
    double Dz_end = -0.00070;
    double dDz = 0.00005;
    int num_dz = (int)round((Dz_end - Dz_start) / dDz) + 1; // 1 Dz value

    printf("Starting OpenMP Parallel execution loops...\n");
    printf("Total tasks: %d (T steps: %d, Dz steps: %d) running on %d threads\n",
        num_T * num_dz, num_T, num_dz, omp_get_max_threads());

    // The collapse(2) pragma allows OpenMP to flatten the nested loops into one 
    // large pool of independent threads, distributing the processing perfectly.
#pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int t_idx = 0; t_idx < num_T; t_idx++) {
        for (int dz_idx = 0; dz_idx < num_dz; dz_idx++) {

            double current_T = T_values[t_idx];
            double current_Dz = Dz_start + dz_idx * dDz;

            printf("Thread %d starting task: T = %.1f K, Dz = %.5f\n",
                omp_get_thread_num(), current_T, current_Dz);

            run_hysteresis_task(current_T, current_Dz);

            printf("Thread %d finished task: T = %.1f K, Dz = %.5f\n",
                omp_get_thread_num(), current_T, current_Dz);
        }
    }

    printf("All batch tasks completed successfully.\n");
    return 0;
}