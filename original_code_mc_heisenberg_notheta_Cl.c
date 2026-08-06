#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

// ============================================================
// 0) Parameters
// ============================================================

#define L 10
#define N (4 * L * L * L)

static const double kB_meV = 8.617e-2; // meV/K
static const double S_mag  = 2.5;      // 5/2

// SI constants for susceptibility
static const double mu0_SI = 4.0 * M_PI * 1e-7;   // H/m
static const double kB_SI  = 1.380649e-23;        // J/K
static const double muB_SI = 9.2740100783e-24;    // J/T
static const double g_fac  = 2.0;

// Lattice (Angstrom)
static const double a = 13.82500;
static const double b =  9.91800;
static const double c =  7.10000;
static double A[3][3];
static double Vcell_A3;
static double V_SI;

// Fractional basis
static double basis_frac[4][3];

// Exchange couplings (meV) – your current Cl values
static const double J1 = -0.296;
static const double J2 = -0.342;
static const double J3 = -0.347;
static const double J4 = -0.310;
static const double J5 = -0.143;

// Single-ion anisotropy (easy axis x)
static const double Dx = 0.04, Dy = 0.0, Dz = 0.0;

// Spins and positions
static double position[N][3];
static double spin_arr[N][3];

// Neighbor lists
static int NNJ1[N];
static int NNJ2[N][2];
static int NNJ3[N];
static int NNJ4[N][2];
static int NNJ5[N][2];

// RNG
static gsl_rng *rng = NULL;

// ============================================================
// Utilities
// ============================================================

static double vec_dot(const double a[3], const double b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static double vec_norm(const double v[3]) {
    return sqrt(vec_dot(v, v));
}

static void vec_copy(double dst[3], const double src[3]) {
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
}

static void vec_cross(double out[3], const double a[3], const double b[3]) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

// ============================================================
// 1) Lattice & basis
// ============================================================

static void init_lattice_constants(void) {
    A[0][0] = a; A[0][1] = 0.0; A[0][2] = 0.0;
    A[1][0] = 0.0; A[1][1] = b; A[1][2] = 0.0;
    A[2][0] = 0.0; A[2][1] = 0.0; A[2][2] = c;
    Vcell_A3 = a * b * c;
    V_SI = (double)(L*L*L) * Vcell_A3 * 1e-30;
}

static double frac_mod(double x) {
    double r = fmod(x, 1.0);
    if (r < 0.0) r += 1.0;
    return r;
}

static void init_basis(void) {
    const double x = 0.1175;
    const double z = 0.31024;
    double r1[3] = { frac_mod(x),      frac_mod(0.25), frac_mod(z)      };
    double r2[3] = { frac_mod(-x+0.5), frac_mod(0.75), frac_mod(z+0.5)  };
    double r3[3] = { frac_mod(-x),     frac_mod(0.75), frac_mod(-z)     };
    double r4[3] = { frac_mod(x+0.5),  frac_mod(0.25), frac_mod(-z+0.5) };
    for (int i = 0; i < 3; i++) {
        basis_frac[0][i] = r1[i];
        basis_frac[1][i] = r2[i];
        basis_frac[2][i] = r3[i];
        basis_frac[3][i] = r4[i];
    }
}

// ============================================================
// 2) Index conversions
// ============================================================

static int coords_to_index(int ix, int iy, int iz, int s) {
    int cell_index = ix * L * L + iy * L + iz;
    return 4 * cell_index + s;
}

static void index_to_coords(int idx, int *ix, int *iy, int *iz, int *s) {
    int cell_index = idx / 4;
    *s  = idx % 4;
    *iz = cell_index % L;
    *iy = (cell_index / L) % L;
    *ix = cell_index / (L * L);
}

// ============================================================
// 3) Spin initialisation & lattice
// ============================================================

static void random_unit_vector(double v[3]) {
    v[0] = gsl_ran_gaussian(rng, 1.0);
    v[1] = gsl_ran_gaussian(rng, 1.0);
    v[2] = gsl_ran_gaussian(rng, 1.0);
    double n = vec_norm(v);
    if (n == 0.0) { v[0] = 1.0; v[1] = 0.0; v[2] = 0.0; }
    else { v[0] /= n; v[1] /= n; v[2] /= n; }
}

static void set_random_spin(int k) {
    double u[3];
    random_unit_vector(u);
    spin_arr[k][0] = S_mag * u[0];
    spin_arr[k][1] = S_mag * u[1];
    spin_arr[k][2] = S_mag * u[2];
}

static void gen_lattice(void) {
    int k = 0;
    for (int ix = 0; ix < L; ix++) {
        for (int iy = 0; iy < L; iy++) {
            for (int iz = 0; iz < L; iz++) {
                double shift[3] = { (double)ix, (double)iy, (double)iz };
                for (int s = 0; s < 4; s++) {
                    set_random_spin(k);
                    double rf[3];
                    rf[0] = basis_frac[s][0] + shift[0];
                    rf[1] = basis_frac[s][1] + shift[1];
                    rf[2] = basis_frac[s][2] + shift[2];
                    position[k][0] = A[0][0]*rf[0] + A[0][1]*rf[1] + A[0][2]*rf[2];
                    position[k][1] = A[1][0]*rf[0] + A[1][1]*rf[1] + A[1][2]*rf[2];
                    position[k][2] = A[2][0]*rf[0] + A[2][1]*rf[1] + A[2][2]*rf[2];
                    k++;
                }
            }
        }
    }
}

// ============================================================
// 4) Neighbours J1..J5
// ============================================================

static void get_nn_J1(void) {
    for (int i = 0; i < N; i++) {
        int ix, iy, iz, s;
        index_to_coords(i, &ix, &iy, &iz, &s);
        if      (s == 1) { NNJ1[i] = coords_to_index(ix, iy, iz, 2); }
        else if (s == 2) { NNJ1[i] = coords_to_index(ix, iy, iz, 1); }
        else if (s == 0) { NNJ1[i] = coords_to_index((ix - 1 + L) % L, iy, iz, 3); }
        else             { NNJ1[i] = coords_to_index((ix + 1) % L, iy, iz, 0); }
    }
}

static void get_nn_J2(void) {
    for (int i = 0; i < N; i++) {
        int ix, iy, iz, s;
        index_to_coords(i, &ix, &iy, &iz, &s);
        int partner, iz2;
        if      (s == 0) { partner = 1; iz2 = (iz + 1) % L; }
        else if (s == 1) { partner = 0; iz2 = (iz - 1 + L) % L; }
        else if (s == 2) { partner = 3; iz2 = (iz + 1) % L; }
        else             { partner = 2; iz2 = (iz - 1 + L) % L; }
        NNJ2[i][0] = coords_to_index(ix, iy, iz,  partner);
        NNJ2[i][1] = coords_to_index(ix, iy, iz2, partner);
    }
}

static void get_nn_J3(void) {
    for (int i = 0; i < N; i++) {
        int ix, iy, iz, s;
        index_to_coords(i, &ix, &iy, &iz, &s);
        if      (s == 2) { NNJ3[i] = coords_to_index(ix, iy, (iz + 1) % L, 1); }
        else if (s == 1) { NNJ3[i] = coords_to_index(ix, iy, (iz - 1 + L) % L, 2); }
        else if (s == 0) { NNJ3[i] = coords_to_index((ix - 1 + L) % L, iy, (iz + 1) % L, 3); }
        else             { NNJ3[i] = coords_to_index((ix + 1) % L, iy, (iz - 1 + L) % L, 0); }
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

// ============================================================
// 5) Energies
// ============================================================

static double local_energy(int k) {
    double E = 0.0;
    int j;

    j = NNJ1[k];    E += -J1 * vec_dot(spin_arr[k], spin_arr[j]);
    j = NNJ2[k][0]; E += -J2 * vec_dot(spin_arr[k], spin_arr[j]);
    j = NNJ2[k][1]; E += -J2 * vec_dot(spin_arr[k], spin_arr[j]);
    j = NNJ3[k];    E += -J3 * vec_dot(spin_arr[k], spin_arr[j]);
    j = NNJ4[k][0]; E += -J4 * vec_dot(spin_arr[k], spin_arr[j]);
    j = NNJ4[k][1]; E += -J4 * vec_dot(spin_arr[k], spin_arr[j]);
    j = NNJ5[k][0]; E += -J5 * vec_dot(spin_arr[k], spin_arr[j]);
    j = NNJ5[k][1]; E += -J5 * vec_dot(spin_arr[k], spin_arr[j]);

    double sx = spin_arr[k][0], sy = spin_arr[k][1], sz = spin_arr[k][2];
    E += -(Dx * sx*sx + Dy * sy*sy + Dz * sz*sz);

    return E;
}

static double total_energy(void) {
    double E = 0.0;
    for (int k = 0; k < N; k++) E += local_energy(k);
    return 0.5 * E; // avoid double counting
}

// ============================================================
// 6) Metropolis — random spin proposal (no theta_max)
// ============================================================

static void propose_random_spin(double out[3]) {
    random_unit_vector(out);
    out[0] *= S_mag;
    out[1] *= S_mag;
    out[2] *= S_mag;
}

static void metropolis_move(double beta) {
    int k = (int)(gsl_rng_uniform(rng) * N);
    if (k == N) k = N - 1;

    double E_old = local_energy(k);
    double old_spin[3];
    vec_copy(old_spin, spin_arr[k]);

    double new_spin[3];
    propose_random_spin(new_spin);
    vec_copy(spin_arr[k], new_spin);

    double dE = local_energy(k) - E_old;
    if (dE > 0.0 && gsl_rng_uniform(rng) >= exp(-beta * dE))
        vec_copy(spin_arr[k], old_spin); // reject
}

static void mcs_sweep(double beta) {
    for (int i = 0; i < N; i++) metropolis_move(beta);
}

// ============================================================
// 7) Magnetization
// ============================================================

static void magnetization_total(double *Mx, double *My, double *Mz) {
    double mx = 0.0, my = 0.0, mz = 0.0;
    for (int i = 0; i < N; i++) {
        mx += spin_arr[i][0];
        my += spin_arr[i][1];
        mz += spin_arr[i][2];
    }
    *Mx = mx; *My = my; *Mz = mz;
}

// ============================================================
// 8) Temperature scan: E(T), Cv(T), chi(T)
// ============================================================

static void temperature_scan(void) {
    const double Tmax = 25.0;
    const double Tmin =  2.5;
    const double dT   =  0.5;

    /*
    // STEP 1: heat from 2K to 30K (no data collection) (Y: skipped that)
    printf("# Heating from 2K to 30K...\n");
    for (double T = 2.0; T <= Tmax; T += 1.0) {
        double beta = 1.0 / (kB_meV * T);
        for (int i = 0; i < 2000; i++) mcs_sweep(beta);
        if ((int)T % 5 == 0) printf("# Heating at T = %.1f K\n", T);
    }
    printf("# Heating complete. Starting cooling scan...\n");
    printf("# T(K) E/N(meV) Cv/N(meV/K) chi_x chi_y chi_z\n");
    */
    

    // STEP 2: cool from 30K to 2K (collect data)
    for (double T = Tmax; T >= Tmin - 1e-9; T -= dT) {
        int n_eq, n_meas, sample_every;
        double beta = 1.0 / (kB_meV * T);

        // Uniform sweeps for all temperatures
        n_eq         = 10000;
        n_meas       = 8000;
        sample_every = 10;

        // Equilibration
        for (int i = 0; i < n_eq; i++) mcs_sweep(beta);

        // Measurement
        double E_sum = 0.0, E2_sum = 0.0;
        double Mx_sum = 0.0, Mx2_sum = 0.0;
        double My_sum = 0.0, My2_sum = 0.0;
        double Mz_sum = 0.0, Mz2_sum = 0.0;

        for (int m = 0; m < n_meas; m++) {
            for (int s = 0; s < sample_every; s++) mcs_sweep(beta);
            double E = total_energy();
            E_sum += E; E2_sum += E * E;
            double Mx, My, Mz;
            magnetization_total(&Mx, &My, &Mz);
            Mx_sum += Mx; Mx2_sum += Mx * Mx;
            My_sum += My; My2_sum += My * My;
            Mz_sum += Mz; Mz2_sum += Mz * Mz;
        }

        double E_mean  = E_sum  / n_meas;
        double E2_mean = E2_sum / n_meas;
        double Mx_mean = Mx_sum / n_meas, Mx2_mean = Mx2_sum / n_meas;
        double My_mean = My_sum / n_meas, My2_mean = My2_sum / n_meas;
        double Mz_mean = Mz_sum / n_meas, Mz2_mean = Mz2_sum / n_meas;

        double Cv_total    = kB_meV * beta * beta * (E2_mean - E_mean * E_mean);
        double Cv_per_ion  = Cv_total / (double)N;
        double chi_x       = beta * (Mx2_mean - Mx_mean * Mx_mean);
        double chi_y       = beta * (My2_mean - My_mean * My_mean);
        double chi_z       = beta * (Mz2_mean - Mz_mean * Mz_mean);

        printf("%6.2f % .8e % .8e % .8e % .8e % .8e\n",
               T, E_mean / (double)N, Cv_per_ion, chi_x, chi_y, chi_z);
        fflush(stdout);
    }
}

static void reinitialize_random_spins(void) {
    for (int i = 0; i < N; i++) set_random_spin(i);
}

// ============================================================
// 9) Thermalization plot: E/N vs sweep number at fixed T
// ============================================================

static void run_thermalization_at_T(double Tfix, const char *filename) {
    int n_eq   = 10000;
    int n_meas = 8000;

    double beta = 1.0 / (kB_meV * Tfix);

    FILE *f = fopen(filename, "w");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); return; }
    fprintf(f, "# step  phase  E_perN[meV]\n");
    // phase=0: thermalization, phase=1: measurement

    reinitialize_random_spins();

    // Equilibration phase
    for (int i = 0; i <= n_eq; i++) {
        double E = total_energy() / (double)N;
        fprintf(f, "%d  0  %.8e\n", i, E);
        mcs_sweep(beta);
    }

    // Measurement phase
    for (int i = 0; i < n_meas; i++) {
        mcs_sweep(beta);
        double E = total_energy() / (double)N;
        fprintf(f, "%d  1  %.8e\n", n_eq + 1 + i, E);
    }

    fclose(f);
}



// ============================================================
// 10) Fluctuations at fixed T
// ============================================================

static void run_fluctuations_C(void) {
    const double Tfix      = 2.0;
    const int n_eq_fluct   = 100000;
    const int n_samples    = 5000;
    const int sample_every = 20;

    double beta = 1.0 / (kB_meV * Tfix);

    for (int i = 0; i < n_eq_fluct; i++) mcs_sweep(beta);

    FILE *f = fopen("fluctuations_T2K_Cl.dat", "w");
    if (!f) { fprintf(stderr, "Cannot open fluctuations file\n"); return; }
    fprintf(f, "# step E_perN[meV] m_x m_y m_z\n");

    for (int t = 0; t < n_samples; t++) {
        for (int s = 0; s < sample_every; s++) mcs_sweep(beta);
        double E = total_energy() / (double)N;
        double Mx, My, Mz;
        magnetization_total(&Mx, &My, &Mz);
        fprintf(f, "%d %.8e %.8e %.8e %.8e\n",
                t, E, Mx/(double)N, My/(double)N, Mz/(double)N);
    }
    fclose(f);
}

// ============================================================
// 11) Main
// ============================================================

int main(void) {
    gsl_rng_env_setup();
    rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, 0UL);

    init_lattice_constants();
    init_basis();
    gen_lattice();
    build_neighbors();

    // Save initial spins
    FILE *f = fopen("spins_before_Cl.txt", "w");
    if (f) {
        for (int i = 0; i < N; i++)
            fprintf(f, "%f %f %f\n", spin_arr[i][0], spin_arr[i][1], spin_arr[i][2]);
        fclose(f);
    }

    // Thermalization plots at three representative temperatures
    run_thermalization_at_T(2.0,  "therm_T2K_Cl.dat");
    run_thermalization_at_T(8.5,  "therm_T8p5K_Cl.dat");  // near T_N^MC (Br)
    run_thermalization_at_T(20.0, "therm_T20K_Cl.dat");

    // Full temperature scan
    temperature_scan();

    // Fluctuations at T = 2 K
    run_fluctuations_C();

    // Save final spins
    f = fopen("spins_after_Cl.txt", "w");
    if (f) {
        for (int i = 0; i < N; i++)
            fprintf(f, "%f %f %f\n", spin_arr[i][0], spin_arr[i][1], spin_arr[i][2]);
        fclose(f);
    }

    gsl_rng_free(rng);
    return 0;
}
