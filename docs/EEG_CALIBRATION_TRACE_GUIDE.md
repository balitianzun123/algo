# EEG Calibration Trace Guide

This document explains the calibration trace ledger. Synthetic examples are
format previews only; real values come from the current EEG calibration run.

## CAL#xx

`CAL#xx` is one actual 4-second Rest trial attempt.

It shows:

- trial time range in the calibration stream
- quality result and bad channels
- ACCEPT or REJECT decision
- accepted count progress

MAIN trace keeps this as one line. Six-channel Rest power is shown once later
in `REST_POWER_MATRIX`. `EEG_TRACE_DETAIL=True` can show the per-trial power
block beside each `CAL#xx`.

## REST_POWER_MATRIX

`[EEG_TRACE][CALIBRATION][REST_POWER_MATRIX]` is the source table for the
accepted Rest trials used by calibration.

Rows are accepted Rest trials, such as `R01` to `R10`. Columns are the action
ROI channel powers. These power values are the input evidence for all later
baseline, ERD, LOO score, and threshold calculations.

## LOO Rxx

`[EEG_TRACE][CALIBRATION][LOO][Rxx]` explains one leave-one-out score.

For `R01`, `R01` is the holdout trial, and the other accepted Rest trials build
the temporary LOO baseline. The trace then shows:

- holdout trial
- reference trials
- LOO baseline per channel in the channel table
- holdout power to ERD in the channel table
- ERD weighted contribution
- Left ROI score
- Right ROI score
- final LOO Action Score as `max(left, right)`

MAIN trace is one compact table per LOO score. It does not repeat the full
reference power list because that data is already in `REST_POWER_MATRIX`.
`EEG_TRACE_DETAIL=True` shows the full reference powers, median values, and
per-channel formula substitutions.

## LOO Baseline

`LOO Baseline` is temporary. It is built only from the reference Rest trials for
one holdout score.

Example: `LOO BASELINE FOR R01` excludes `R01` and uses `R02` through `R10`.

## T_all

`[EEG_TRACE][CALIBRATION][THRESHOLD][T_ALL]` explains the normal six-channel
calibration threshold.

MAIN trace shows a compact, still-recomputable Robust Median + MAD block:

- original ten LOO Action Scores
- sorted scores
- `method = robust_median_mad`
- `center = median(scores)`
- `MAD = median(abs(scores - center))`
- `robust_sigma = 1.4826 * MAD`
- `robust_upper = center + k * robust_sigma`
- `max_rest_score`
- final threshold

`EEG_TRACE_DETAIL=True` also shows each sorted index.

The final threshold comes from `max(max_rest_score, robust_upper)`, using the
same robust threshold calculation used by the algorithm.

## T_no_xxx

`[EEG_TRACE][CALIBRATION][SINGLE_CHANNEL_THRESHOLDS]` summarizes the six
single-channel exclusion thresholds.

MAIN trace shows:

- excluded channel
- left and right ROI weight sums after exclusion
- robust MAD threshold
- ten LOO scores recomputed without that channel
- final `T_no_xxx`

These thresholds are used by prediction only when exactly one action channel is
excluded by quality.

`EEG_TRACE_DETAIL=True` keeps the full per-channel `T_NO_xxx` robust threshold
derivation and optional exclusion LOO blocks.

## FINAL PERSONAL BASELINE

`[EEG_TRACE][CALIBRATION][FINAL_BASELINE]` explains the baseline saved into the
personal profile and later used by prediction.

For each channel, it shows:

- the actual aggregation rule, currently `median`
- the middle value or values that define the median
- final channel baseline

The full ten accepted Rest powers are not repeated in MAIN because they are
already in `REST_POWER_MATRIX`. `EEG_TRACE_DETAIL=True` shows the complete
per-channel Rest powers and sorted powers.

Important: `LOO Baseline != Final Personal Baseline`.

LOO baselines are temporary holdout baselines used to generate Rest LOO scores.
The final personal baseline is built from all accepted Rest trials and saved in
the profile for prediction.
