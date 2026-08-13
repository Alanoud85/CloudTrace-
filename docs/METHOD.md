# Method summary

## 1. Curation and sessions

1. Parse event timestamps in UTC.
2. Remove invalid timestamps.
3. Remove repeated event identifiers.
4. Construct an identity key from the most specific available principal field.
5. Sort within identity by event time.
6. Start a new session after more than 30 minutes of inactivity.
7. Sort sessions globally by start time.
8. Assign 70% of complete sessions to training, 15% to validation, and the remainder to the future test period.

No rarity, transition, scaler, or identity-relation reference is fitted before the chronological split.

## 2. Session views

The implementation creates the 80 features recorded in the archived run manifest:

- 47 event/content features
- 21 temporal/behavioral features
- 12 identity/relation features

Top action and service proportions are learned from training events. Rarity scores use negative log training probabilities with an empirical floor. Transition surprise uses training action-transition probabilities. Identity novelty compares observed session relationships with train-period identity reference sets.

## 3. Network

Each view uses:

```text
Linear(input, 96) -> GELU -> LayerNorm(96) -> Linear(96, 32) -> LayerNorm(32)
```

The three 32-dimensional latent vectors are concatenated and passed through a gate:

```text
Linear(96, 96) -> GELU -> Linear(96, number_of_views) -> Softmax
```

The gate produces session-specific nonnegative weights summing to one. Their weighted latent sum is decoded independently back into each view.

## 4. Training objective

Training adds random feature masking and Gaussian perturbation to the inputs. The loss combines:

- mean view reconstruction error;
- pairwise latent agreement;
- a variance guard that penalizes collapsed latent dimensions.

Optimization uses AdamW, a batch size of 1,024, at most 18 epochs, and early stopping with patience 4.

## 5. Anomaly score

The score uses three view reconstruction errors and one latent disagreement component. Components are robustly centered and scaled on validation data. The final weights are 2/7 for each reconstruction component and 1/7 for disagreement, matching the archived seed-42 checkpoint.

## 6. EVT threshold

The 0.98 validation-score quantile defines the peaks-over-threshold tail. Excesses are fitted with a generalized Pareto distribution. For a target alert probability `alpha`, the score threshold is obtained from the conditional tail probability `alpha / (1 - q)`.

## 7. Controlled perturbations

Five controlled families probe score sensitivity: burst, context hijack, API sequence, failure storm, and region/service switch. Their parameters are fixed in `configs/paper.yaml`. They are benchmark constructs and are not treated as verified attacks.
