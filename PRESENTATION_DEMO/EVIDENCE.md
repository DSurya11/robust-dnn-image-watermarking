================================================================
IMPLEMENTATION EVIDENCE (40% of marks)
================================================================

For each item below, check the paper AND the code, then mark:
[CONFIRMED] = paper says it, code does it, they match
[PARTIAL]   = paper says it, code has it but not fully working
[MISSING]   = paper says it, code does NOT have it
[EXTRA]     = code has it, paper does not mention it

--- ARCHITECTURE COMPONENTS ---

1. ISN (Invertible Steganography Network) as the backbone
   Paper reference: Section 3.1, Figure 1 ("ISN" block in Generator)
   Code location: models/generator.py → SimpleISN class
   Status: [PARTIAL]
   Note: Paper describes an invertible neural network (INN) ISN backbone.
         Our SimpleISN is an encoder-decoder CNN (Conv/ReLU stack) with
         embed() producing a small perturbation and extract() decoding.
         No coupling layers / explicit invertibility constraints found.

2. GAN framework (Generator + Discriminator adversarial training)
   Paper reference: Section 3.2, Equations 4 and 5
   Code location: models/discriminator.py, train.py Phase 3
   Status: [PARTIAL]
   Note: Discriminator exists (CNN binary classifier), and utils/losses.py
         defines GAN-style L_D / L_G / L_adv. However, train.py Phase 3
         does not instantiate or train the discriminator and uses only MSE
         losses. Final evaluation uses Phase 1 checkpoint, not GAN training.

3. Dynamic Attention Enhancement Module (Swin Transformer + DenseNet)
   Paper reference: Section 3.3, Figure 2 and Figure 3
   Code location: models/enhance.py
   Status: [PARTIAL]
   Note: EnhanceModule contains DenseBlock(s) and a SimpleSwinBlock. But the
         Swin component is not true window attention in this repo because
         timm is not installed; the code falls back to global MultiheadAttention
         over flattened H*W tokens. Dynamic MLP weighting exists, but the
         SwinBlock + window attention behavior from the paper is approximated.

4. Differential Feature Extraction Module (4-layer DenseNet)
   Paper reference: Section 3.4, Figure 4
   Code location: models/feat_extract.py
   Status: [CONFIRMED]
   Note: DifferentialFeatureExtractor.forward(watermarked, attacked) takes BOTH
         x_c and x_d (same shapes), forms diff = watermarked - attacked, and
         runs 4 conv layers with dense concatenation before projecting back to 3
         channels (differential features).

5. Pre-enhancement and Post-enhancement sub-modules
   Paper reference: Section 3.3 ("pre-enhancement emphasizes global noise,
                    post-enhancement targets local artifacts")
   Code location: models/enhance.py
   Status: [PARTIAL]
   Note: train.py Phase 3 constructs enhance_pre and enhance_post as two separate
         EnhanceModule instances. However, both are created with the same default
         parameters (including same window_size default=4), and there is no
         explicit small-window (4×4) vs large-window (8×8) split implemented.

6. Three-phase training strategy
   Paper reference: Section 4.1 Implementation Details ("1600 epochs, phased training")
   Code location: train.py
   Status: [CONFIRMED]
   Note: train.py supports --phase 1/2/3 with different training routines:
         Phase 1 (clean embed/extract), Phase 2 (robustness via attacks),
         Phase 3 (adds Enhance + FeatExtract stack). The exact losses and
         training details differ from the paper (see Differences section).

7. DWT (Discrete Wavelet Transform) low-frequency loss
   Paper reference: Section 3.5, Equation 8 (L_wavelet / low-frequency wavelet loss)
   Code location: utils/losses.py
   Status: [PARTIAL]
   Note: utils/losses.py implements a Haar LL extraction (_haar_ll) and computes
         L_wavelet = MSE(LL(x_c), LL(x_h)). However, this wavelet loss is not
         used anywhere in train.py (compute_losses() is never called).

8. All 5 loss functions from paper
   Paper reference: Section 3.5, Equations 1-10
   L_pre, L_post, L_enhance, L_D, L_G, L_adv, L_f, L_wavelet,
   L_stage, L_total
   Code location: utils/losses.py
   Status: [PARTIAL]
   Note: utils/losses.py provides L_pre, L_post, L_enhance, L_D, L_G, L_adv,
         L_f, L_wavelet, and L_stage in compute_losses(). L_total (Eqn. 10 style
         accumulation) is not implemented as a named function/value, and none of
         these losses are wired into train.py (train.py uses hand-written MSE
         losses per phase).

9. Attack simulation during training (Gaussian, JPEG, Round)
   Paper reference: Section 4.1 (distortion attacks to generate attacked images)
   Code location: train.py (apply_random_attack), experiments/run_forward.py, utils/attacks.py
   Status: [CONFIRMED]
   Note: Phase 2/3 training applies random distortion attacks (Gaussian, JPEG,
         Round) to generate x_d from x_c. The attack implementation in train.py
         is separate from utils/attacks.py but matches the same attack types.

10. Evaluation metrics PSNR-C and PSNR-S
    Paper reference: Section 4.1 Metrics
    Code location: evaluate.py
    Status: [CONFIRMED]
    Note: evaluate.py computes PSNR-C = PSNR(x_h, x_c) and PSNR-S = PSNR(x_s, x_e)
          and reports them for Gaussian/JPEG/Round attacks.

--- DATASETS ---

11. DIV2K dataset (800 training, 100 val, 100 test, 224x224 patches)
    Paper reference: Section 4.1 Dataset
    Our implementation: [verify — what dataset and size do we use?
                        note the difference honestly]
    Status: [MISSING]
    Note: Code trains/evaluates on local images in data/ (≈80 images → 79 pairs)
          resized/cropped to 128×128 (train.py, evaluate.py, data/prepare_data.py).
          No DIV2K download/loader pipeline is present.

--- EVALUATION LEVELS ---

12. Level 1 evaluation (single attack type training and testing)
    Paper reference: Section 4.2.1, Table 1
    Code location: evaluate.py
    Status: [PARTIAL]
    Note: evaluate.py matches the paper’s Level-1 attack suite (Gaussian σ=1/10,
          JPEG QF=90/80, Round) and reports PSNR-C/PSNR-S. However, the paper’s
          multi-level evaluation/training protocol (Level 1–4 definitions) is not
          implemented as explicit levels in this codebase, and our training does
          not follow “single-attack training” vs “multi-attack training” levels.

================================================================
MODIFICATION EVIDENCE (60% of marks)
================================================================

For each modification, confirm the following exist:
[CODE] = the actual code change is in the codebase
[DATA] = a results CSV exists in results/final/
[CHART] = a chart PNG exists in results/final/charts/ or similar
[INTERPRETED] = there is a written interpretation (in INSTRUCTIONS.txt)

MOD1 — Geometric Attack Robustness (NOT in paper)
  Paper does NOT test: rotation, crop, brightness attacks
  Code change: utils/attacks.py → rotation_attack(), crop_attack(),
               brightness_attack() functions
  [CODE]: CONFIRMED (functions exist in utils/attacks.py)
  [DATA]: CONFIRMED (results/final/mod1_geometric_results.csv)
  [CHART]: CONFIRMED (results/final/mod1_geometric_chart.png)
  [INTERPRETED]: CONFIRMED (PRESENTATION_DEMO/INSTRUCTIONS.txt MOD1)
  Key finding to state: "Brightness mild=18.74 dB (robust),
  Strong rotation=11.74 dB (not robust) — geometric robustness
  is a limitation of the current model."

MOD2 — Lambda Hyperparameter Tradeoff Analysis (NOT in paper)
  Paper sets lambda_c=1.0, lambda_s=1.0 without analysis
  Code change: experiments/mod2_lambda_tuning.py →
               fine-tunes model for 15 epochs per lambda config
  [CODE]: CONFIRMED (fine-tuning loop exists, not just evaluation)
  [DATA]: CONFIRMED (results/final/mod2_lambda_results.csv)
  [CHART]: CONFIRMED (results/final/mod2_lambda_chart.png)
  [INTERPRETED]: CONFIRMED (PRESENTATION_DEMO/INSTRUCTIONS.txt MOD2)
  Key finding to state: "As lambda_c increases from 0.3→1.7,
  PSNR-C rises 24.92→28.37 dB, PSNR-S drops 18.64→17.78 dB.
  Clear quality tradeoff confirmed."

MOD3 — Secret Image Content Type Analysis (NOT in paper)
  Paper only uses natural photos as secret images
  Code change: experiments/mod3_capacity_analysis.py →
               generates Noise/Text/Logo/Natural secret types
  [CODE]: CONFIRMED (4 secret types generated programmatically)
  [DATA]: CONFIRMED (results/final/mod3_capacity_results.csv)
  [CHART]: CONFIRMED (results/final/mod3_capacity_chart.png)
  [INTERPRETED]: CONFIRMED (PRESENTATION_DEMO/INSTRUCTIONS.txt MOD3)
  Key finding: "Text secrets extract best (PSNR-S=13.4, SSIM=0.138).
  Noise secrets are hardest (PSNR-S=6.8, SSIM=0.005)."

MOD4 — Data Augmentation to Reduce Overfitting (NOT in paper)
  Paper does not address overfitting on small datasets
  Code change: train.py or utils/attacks.py →
               training_augment() with flip + brightness jitter
  [CODE]: CONFIRMED (train.py uses RandomFlip/ColorJitter; optional training_augment())
  [DATA]: CONFIRMED (results/final/mod4_augmentation_results.csv)
  [CHART]: MISSING (no mod4 chart PNG found under results/final)
  [INTERPRETED]: CONFIRMED (PRESENTATION_DEMO/INSTRUCTIONS.txt MOD4)
  Key finding: "Train/val gap reduced from 1.00→0.39 dB (61% reduction)
  showing augmentation successfully reduces overfitting."

MOD5 — Differentiable SSIM Loss (NOT in paper)
  Paper uses only MSE (L2 norm) for all losses
  Code change: utils/losses.py → ssim_loss() using pure PyTorch
               Gaussian kernel convolution (fully differentiable)
  [CODE]: CONFIRMED (ssim_loss uses F.conv2d, not skimage)
  [DATA]: CONFIRMED (results/final/mod5_ssim_results.csv)
  [CHART]: MISSING (no mod5 chart PNG found under results/final)
  [INTERPRETED]: CONFIRMED (PRESENTATION_DEMO/INSTRUCTIONS.txt MOD5)
  Key finding: "Hybrid MSE+SSIM trades PSNR-S -0.62 dB for SSIM
  +0.081 improvement. Perceptual vs fidelity tradeoff demonstrated."

================================================================
HONEST DIFFERENCES FROM PAPER (state these in your report)
================================================================

List every confirmed difference between paper and implementation.
For each, write one sentence explaining why it differs.

CONFIRMED differences (I am sure about these — verify and keep):

D1. SimpleISN encoder-decoder vs true INN coupling layers
    "Due to implementation complexity of invertible coupling layers,
     we use a CNN encoder-decoder that achieves embedding/extraction
     without strict mathematical invertibility."

D2. Phase 1 checkpoint used for final evaluation vs full pipeline
    "The GAN and attention components are implemented but the final
     evaluated model uses Phase 1 training. Full pipeline training
     exceeded available compute resources (CPU only, no GPU)."

D3. 79 synthetic images vs 800 DIV2K natural images
    "Dataset size limited by available images in the provided data/ folder
     and kept small for fast CPU experimentation."

D4. ~200 training epochs vs 1600 epochs in paper
    "Training time constraint on CPU hardware."

D5. 128x128 input size vs 224x224 in paper
    "Reduced to 128x128 for feasible CPU training time and memory."

UNCERTAIN differences (agent must verify by reading code):

U1. Does enhance.py use true Swin Transformer window attention
    or a simplified approximation?
    → Verified: timm is not installed; EnhanceModule falls back to global
      MultiheadAttention over flattened tokens (not window attention).

U2. Does the discriminator in models/discriminator.py match the
    multi-level steganalysis structure from paper Section 3.2?
    → Verified: discriminator is a 4-block Conv/BN/LeakyReLU/MaxPool CNN with
      spectral norm and an MLP head; no explicit multi-level steganalysis
      structure from the paper is implemented.

U3. Is the DWT wavelet loss (Equation 8) actually computed and
    added to the training loss in train.py?
    → Verified: wavelet loss is implemented in utils/losses.py (Haar LL),
      but train.py never calls compute_losses(), so it is not used.

U4. In feat_extract.py, does the forward pass take BOTH xc
    (watermarked) and xd (attacked) as separate inputs,
    or just one image?
    → Verified: forward(watermarked, attacked) takes both tensors and uses
      their difference to produce differential features.

================================================================
FINAL SUMMARY TABLE
================================================================

After completing all checks above, generate this table:

| Component              | Paper | Code | Match Level  |
|------------------------|-------|------|--------------|
| ISN backbone           | Yes   | Yes  | Partial      |
| GAN framework          | Yes   | Yes  | Partial      |
| Dynamic attention      | Yes   | Yes  | Partial      |
| Differential features  | Yes   | Yes  | Confirmed    |
| 3-phase training       | Yes   | Yes  | Confirmed    |
| All loss functions     | Yes   | Yes  | Partial      |
| Attack simulation      | Yes   | Yes  | Confirmed    |
| PSNR-C/S evaluation    | Yes   | Yes  | Confirmed    |
| MOD1 geometric attacks | No    | Yes  | Extra (our mod)|
| MOD2 lambda tuning     | No    | Yes  | Extra (our mod)|
| MOD3 content types     | No    | Yes  | Extra (our mod)|
| MOD4 augmentation      | No    | Yes  | Extra (our mod)|
| MOD5 SSIM loss         | No    | Yes  | Extra (our mod)|
