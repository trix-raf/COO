Dari hasil pemindaian, seluruh modul aktif (`config.py`, `main.py`, `core/*`, `problems/*`, `utils/*`) sudah bersih dengan ukuran `core/quasi_newton.py` mencapai `39192 B` (yang sudah memuat **Stage-3E VarPro**).

*Catatan kebersihan direktori:* Di root proyek `D:\VSCodeProjects\co` masih terdapat folder cadangan `COCOv2/` dan file `COCOv1.zip`—Anda bisa menghapus keduanya kapan saja setelah `ARCHITECTURE.md` diperbarui karena seluruh kodenya sudah aktif di root proyek.

Berikut adalah isi lengkap **`ARCHITECTURE.md`** terbaru yang sudah disinkronkan 100% dengan struktur repositori dan seluruh komponen algoritma Anda:

```markdown
# COCO — Completely Over Complicated Optimizer (`SwarmAdamBFGS`)
**Peta Arsitektur Modular, Alur Matematis 3-Fase, dan Referensi Cepat Kode**

---

## 1. Struktur Direktori Proyek

```text
co/
├── ARCHITECTURE.md                  # Dokumen navigasi arsitektur & peta modul proyek
├── config.py                        # Pusat pengaturan path output & konfigurasi hierarkis 3-Fase
├── main.py                          # Entry point CLI & runner benchmark modular (P1 s/d P8)
├── requirements.txt                 # Dependensi proyek (numpy, matplotlib, pillow, imageio-ffmpeg)
│
├── core/                            # Inti mesin optimasi hibrida 3-Fase
│   ├── __init__.py                  # Ekspor publik (SwarmAdamBFGS, ProjectedDampedBFGS, FrozenSubspaceSuperFinetuner)
│   ├── optimizer.py                 # Orkestrator utama SwarmAdamBFGS (Stage 1 -> Stage 2 -> Stage 3)
│   ├── schedules.py                 # Penjadwalan cosine annealing decay & L2 gradient norm clipping
│   ├── sensitivity.py               # Estimasi kelengkungan Hessian, pembobotan, & deteksi sub-ruang lemah
│   ├── stochastic_ops.py            # Latin Hypercube Sampling, pantulan batas, seleksi basin, & hybrid re-heat
│   ├── quasi_newton.py              # Solver Stage-2 (ProjectedDampedBFGS) & Stage-3 (FrozenSubspaceSuperFinetuner + VarPro)
│   └── early_stopping.py            # Weight-gated sliding window memory early stopping & deteksi stagnasi
│
├── problems/                        # Definisi 8 benchmark analitik & neural network (Pure NumPy)
│   ├── __init__.py                  # Auto-discovery & registry otomatis (@register_problem / register_problem)
│   ├── base.py                      # Kontrak dataclass standar ProblemSpec
│   ├── p1_weak_coupling.py          # [P1] 5D Weak-Coupling Double Exponential + High-Frequency Drift
│   ├── p2_rosenbrock_exp.py         # [P2] 3D Coupled Rosenbrock-Exponential + Vanishing Sinus (60.000x ratio)
│   ├── p3_stiff_prony.py            # [P3] 4D Stiff Sum of Exponentials (Fast vs Slow Compartmental Tail)
│   ├── p4_lorentzian_chirp.py       # [P4] 6D Lorentzian Needle Resonance + Quadratic Frequency Chirp
│   ├── p5_numpy_deep_mlp.py         # [P5] 14D Pure-NumPy 2-Hidden-Layer Heterogeneous Deep MLP + Skip
│   ├── p6_compact_mlp.py            # [P6] 7D Pure-NumPy 1-Hidden-Layer Heterogeneous MLP (Diagnostic)
│   ├── p7_hybrid_trancendental.py   # [P7] 14D Hybrid: Cubic Poly + Exp + ln/log10 + Transcendental + Step
│   └── p8_fm_rational_coupled.py    # [P8] 12D Non-Separable FM-Gaussian Wavepacket + Rational + Tanh
│
├── utils/                           # Pelaporan presisi tinggi, grafik statis, & ekspor animasi
│   ├── __init__.py                  # Ekspor publik (emit_report, draw_beta_changes, animate_function_fit)
│   ├── reporting.py                 # Laporan presisi 17-digit float64, galat absolut/relatif, & selisih ULP
│   ├── visualizer.py                # Grafik statis per-parameter (_grid.png) & gabungan (_combined.png)
│   └── animator.py                  # Renderer video MP4 (QSV/NVENC/x264) & GIF berdurasi target 30 detik
│
└── outputs/                         # Direktori artefak hasil eksekusi (otomatis dibuat oleh PathConfig)
    ├── reports/                     # File teks laporan numerik (*_report.txt)
    ├── plots/                       # Gambar konvergensi statis (*_grid.png, *_combined.png)
    └── animations/                  # Animasi kecocokan kurva & lintasan iterasi (*_fit.mp4, *_fit.gif)

```

---

## 2. Alur Eksekusi & Arsitektur 3-Fase (`SwarmAdamBFGS`)

### Fase 1: Eksplorasi Global Stokastik & Orde-1 (`Stage1SwarmConfig`)

* **Inisialisasi Populasi Adaptif (`core/stochastic_ops.py`):** Populasi partikel diskalakan terhadap dimensi (`max(n_particles, min_particles_per_dim * dim)`) dan disebar merata menggunakan *Latin Hypercube Sampling* (`latin_hypercube_sampling`).
* **Dinamika Velocity + Adam (`core/optimizer.py`, `core/schedules.py`):** Setiap partikel memperbarui momentum orde-1 ($\mathbf{m}_i$) dan orde-2 ($\mathbf{s}_i$) dengan *L2 gradient clipping*, digabungkan dengan inersia kelajuan ($\omega_t$) dan laju pembelajaran ($\eta_t$) yang meluruh mengikuti *cosine annealing* (`cosine_decay`).
* **Keputusan Stokastik & Re-Heat (`core/stochastic_ops.py`):**
* Menggunakan sebaran standar ternormalisasi *dimension-invariant* ($\sigma_{\text{RMS}}$) untuk mendeteksi konvergensi dini populasi.
* Jika populasi menyempit terlalu cepat sementara gradien masih besar, `stochastic_reheat` memicu *Hybrid Global-Local Re-Heat* (separuh partikel eksplorasi lokal, separuh memindai cekungan baru secara global).
* `stratified_subspace_research` menyisir ulang dimensi yang memiliki bobot sensitivitas lemah.



### Transisi & Pembobotan Sensitivitas (`SensitivityWeighter`)

* **Estimasi Kelengkungan (`core/sensitivity.py`):** Menghitung kelengkungan lokal tiap parameter ($\kappa_j$) menggunakan beda hingga pusat pada gradien dan pemindaian interval datar (*flat-interval scanning*).
* **Pembobotan Invers (`param_weights`):** Memberikan bobot tinggi pada parameter bergradien landai/lemah ($w_j \propto \kappa_j^{-p}$) dan mempartisi parameter dominan vs lemah (`detect_insensitive_subspace`) menggunakan analisis celah logaritmik adaptif (*adaptive log-gap partitioning*).

### Fase 2: Multi-Basin Multi-Secant BFGS Tournament (`Stage2QuasiNewtonConfig`)

* **Seleksi Cekungan Beragam (`select_diverse_basins`):** Memilih $K$ kandidat `pbest` paling beragam secara spasial dari riwayat swarm.
* **Inisialisasi Hessian Multi-Secant (`build_multisecant_hessian`):** Mengonstruksi aproksimasi matriks Hessian dari pasangan $(\Delta\boldsymbol{\beta}, \Delta\mathbf{g})$ riwayat swarm melalui generalisasi Broyden dengan regularisasi Tikhonov.
* **Turnamen & Pemolesan (`ProjectedDampedBFGS`):** Memoles tiap kandidat selama `multi_basin_warmup_steps` menggunakan proyeksi batas aktif (`_project_gradient`), *Armijo backtracking line search*, dan auto-reset Hessian, lalu melanjutkan kandidat terbaik hingga konvergen.

### Fase 3: Frozen-Subspace Super-Finetuning & VarPro (`Stage3SubspaceConfig`)

Aktif secara otomatis ketika *loss* akhir Fase 2 masih berada di atas `stage3_loss_trigger` (akibat lembah datar berdimensi tinggi atau kolinearitas fungsional):

1. **Stage-3A — Subspace Differential Evolution (`_subspace_differential_evolution`):** Membekukan parameter dominan dan menjalankan evolusi diferensial hibrida (`DE/best/1` & `DE/rand/1`) hanya pada sub-ruang parameter lemah.
2. **Stage-3B — Subspace Local Refinement:** Memoles sub-ruang aktif secara berurutan lewat `_subspace_bfgs`, `_subspace_trust_region_lm`, `_subspace_nelder_mead` (bebas turunan), dan `solve_residual_root_gauss_newton` (SVD Levenberg-Marquardt pada matriks Jacobian $N \times k$ tanpa mengkuadratkan skala parameter mikro).
3. **Stage-3C — Joint Full-Space Polish:** Membuka seluruh dimensi parameter untuk menyelaraskan kopling antar-parameter secara penuh via BFGS, Trust-Region LM, dan Full-Space SVD Residual-Root Gauss-Newton.
4. **Stage-3E — Variable Projection (`_varpro_nonlinear_gauss_newton`):** Mendeteksi otomatis parameter linier vs nonlinier (`_partition_linear_nonlinear`), menyaring sub-ruang nonlinier via *Latin Hypercube Sampling*, lalu menyelesaikan parameter linier secara analitik eksak lewat SVD `lstsq` di dalam evaluasi residu Gauss-Newton—menembus lembah kolinearitas berdimensi tinggi (seperti P7 14D) hingga presisi mesin (`float64`).
5. **Stage-3D — Global Multi-Start LHS Residual-Root Escape:** Cadangan pencarian global multi-start berbasis SVD Gauss-Newton bila residu non-separable masih di atas ambang target.

---

## 3. Panduan Navigasi Cepat Modul (Peta Kelas & Fungsi)

| File | Kelas / Fungsi Utama | Tanggung Jawab |
| --- | --- | --- |
| **`config.py`** | `PathConfig`, `AnimationConfig`, `Stage1SwarmConfig`, `SensitivityConfig`, `Stage2QuasiNewtonConfig`, `Stage3SubspaceConfig`, `EarlyStoppingConfig`, `OptimizerConfig` | Menyimpan seluruh jalur output (`outputs/`) dan hyperparameter 3-Fase secara terstruktur. |
| **`main.py`** | `run_single_problem()`, `main()` | Parser argumen CLI (`-p`, `--no-gif`, `--no-mp4`, `--show`), memuat masalah, menjalankan `SwarmAdamBFGS`, dan memanggil utilitas output. |
| **`core/optimizer.py`** | `SwarmAdamBFGS` | Mengorkestrasi siklus hidup iterasi Fase 1, perpindahan ke Fase 2, evaluasi *early stopping*, dan pemicu Fase 3. |
| **`core/quasi_newton.py`** | `build_multisecant_hessian()`, `solve_residual_root_gauss_newton()`, `ProjectedDampedBFGS`, `FrozenSubspaceSuperFinetuner` | Seluruh solver orde-2 (BFGS, Trust-Region LM, SVD Gauss-Newton), *Nelder-Mead*, *Subspace DE*, dan *VarPro*. |
| **`core/sensitivity.py`** | `SensitivityWeighter` | Menghitung kelengkungan Hessian diagonal, bobot sensitivitas, serta masker sub-ruang lemah (`weak_mask`). |
| **`core/stochastic_ops.py`** | `latin_hypercube_sampling()`, `apply_boundary_reflection()`, `select_diverse_basins()`, `stochastic_reheat()`, `stratified_subspace_research()` | Operator stokastik populasi, pemantulan batas elastis, pemilihan cekungan beragam, dan *re-heat* hibrida. |
| **`core/early_stopping.py`** | `WeightedMemoryEarlyStopper` | Memantau kestabilan riwayat berbobot pada jendela geser (`memory_window`) dan mendeteksi stagnasi di atas target. |
| **`core/schedules.py`** | `cosine_decay()`, `clip_gradient()` | Jadwal peluruhan kosinus ($\eta_t, \omega_t$) dan pemotongan norma gradien. |
| **`problems/__init__.py`** | `register_problem()`, `get_problem()`, `list_problem_codes()` | Pemuatan otomatis seluruh modul `p*_*.py` tanpa pendaftaran manual. |
| **`problems/base.py`** | `ProblemSpec` | Struktur data standar (`code`, `title`, `bounds`, `beta_true`, `loss_fn`, `grad_fn`, `labels`, `x_domain`, `eval_fn`). |
| **`utils/reporting.py`** | `format_high_precision_report()`, `emit_report()` | Mencetak dan menyimpan tabel perbandingan 17-digit `float64`, galat absolut/relatif, dan jarak bit ULP. |
| **`utils/visualizer.py`** | `draw_beta_changes()` | Membuat plot lintasan konvergensi tiap parameter ($\beta_j$) beserta garis referensi nilai sejati. |
| **`utils/animator.py`** | `animate_function_fit()` | Merender animasi evolusi kecocokan kurva $\hat{y}(x)$ terhadap $y(x)$ berdurasi 30 detik dengan akselerasi perangkat keras (QSV/NVENC/OpenCL) jika tersedia. |

---

## 4. Ringkasan 8 Kasus Uji Benchmark (`problems/`)

| Kode | Nama Masalah | Dimensi | Karakteristik & Tantangan Numerik Utama |
| --- | --- | --- | --- |
| **P1** | `Weak-Coupling Double Exponential + Drift` | 5D | Amplitudo mikro ($0.0018$) & peluruhan lambat ($-0.00035$) tersembunyi di bawah eksponen utama. |
| **P2** | `Coupled Rosenbrock-Exp + Vanishing Sinus` | 3D | Lembah melengkung curam dengan rasio sensitivitas parameter ekstrem ($60.000\times$). |
| **P3** | `Stiff Sum of Exponentials (Prony)` | 4D | Pemisahan laju peluruhan cepat ($\beta_2=2.8$) dan ekor kompartemen lambat ($\beta_4=0.015$). |
| **P4** | `Lorentzian Needle + Quadratic Chirp` | 6D | Puncak resonansi jarum sempit (gradien nol di luar puncak) + sinyal *chirp* kuadratik mikro. |
| **P5** | `Pure-NumPy 2-Hidden-Layer Deep MLP + Skip` | 14D | Jaringan saraf 2 lapisan tersembunyi heterogen ($\tanh$, $\sin$, $\text{swish}$) dengan jalur gain mikro ($u_2=0.038$). |
| **P6** | `Pure-NumPy 1-Hidden-Layer Compact MLP` | 7D | Benchmark diagnostik 1 lapisan tersembunyi untuk mengisolasi rasio gain neuron $61.8\times$. |
| **P7** | `14D Hybrid Transcendental + Step` | 14D | Gabungan polinom kubik, eksponen, $\ln$, $\log_{10}$, trigonometri terkopel, dan *smooth step*; menguji kolinearitas Taylor & proyeksi VarPro. |
| **P8** | `12D FM-Gaussian Wavepacket + Rational + Tanh` | 12D | Sistem nonlinier bersarang penuh (*non-separable*) dengan modulasi frekuensi FM dan puncak rasional. |

---

## 5. Cara Penggunaan & Ekstensi

### Menjalankan Benchmark dari Terminal

```bash
# Jalankan seluruh kasus uji (P1 sampai P8)
python main.py -p ALL

# Jalankan satu kasus spesifik (misal P7 atau P8)
python main.py -p P7

# Jalankan tanpa merender GIF/MP4 (eksekusi cepat hanya laporan & plot PNG)
python main.py -p ALL --no-gif --no-mp4

```

### Menambah Kasus Uji Baru (`P9`, dst.)

1. Buat file baru di `problems/p9_nama_kasus.py` yang mengembalikan `ProblemSpec`.
2. Tambahkan `register_problem(build_problem())` di baris paling bawah file tersebut.
3. Jalankan `python main.py -p P9` — `problems/__init__.py` akan menemukannya secara otomatis.

