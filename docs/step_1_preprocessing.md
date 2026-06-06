# Step 1: Data loading and preprocessing

The preprocessing step converts image/CSV pairs into two arrays:

```text
images: (N, 150, 150, 1)
labels: (N, 6, 6, 6)
```

## Important design decisions

1. Images are loaded as grayscale, resized if necessary, converted to `float32`, and divided by 255.
2. CSV values are not normalized by default, because the area fractions were already normalized during dataset preparation.
3. CSV rows are mapped to a `6 × 6` grid. If `x` and `y` exist, the code uses them as spatial coordinates. Otherwise, the row order is used.
4. Pairing is robust to both same-stem pairs and the old notebook naming convention where `P_sub_*.png` corresponds to `A_sub_*.csv`.

## Grid-size note

The original notebook used `grid_size = 24` with `150 × 150` images and later reshaped labels to `6 × 6`. Since `150 / 6 = 25`, this repository uses `grid_shape=(6, 6)` as the primary definition and computes the cell size from the image size. This avoids boundary mismatches.
