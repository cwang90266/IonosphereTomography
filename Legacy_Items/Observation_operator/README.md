# Observation_operator — LOS-only Fibonacci operator

This package leaves `observation_preparation` untouched.

Key behavior:
- Global equal-area Fibonacci mesh uses the existing 5° definition.
- For each RO event, only global Fibonacci triangle vertices actually touched by LOS segment midpoints within the supplied altitude grid are retained as state columns.
- No ROI-disk points are added just for display and no nearest-node fallback invents off-LOS columns.
- The 2000-km circle is observation-selection metadata / a plotting reference. It is **not** used to truncate a physical RO ray. A LOS can extend outside that circle; clipping it would remove real Abel TEC contribution.
- Abel validation ignores `Abel_TEC_cal_TECU` and `Abel_TEC_forward_TECU`; it uses only `Abel_alt_km`, `Abel_Ne`, and the real LEO/GNSS geometry.
- State order is altitude-major and `TEC = H @ Ne_state`.

Run from project root:
`python -m Observation_operator.test_observation_operator`
