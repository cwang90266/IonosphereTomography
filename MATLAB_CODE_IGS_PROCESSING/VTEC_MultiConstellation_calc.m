function [nPRNdataOUT, lPRNdataOUT, gPRNdataOUT, bPRNdataOUT, TECANS, Bias, K_all, DCBcomparison] = ...
    VTEC_MultiConstellation_calc(nPRNdata, lPRNdata, gPRNdata, bPRNdata, ...
                                  nSat, lSat, gSat, bSat, ...
                                  timeGNSS, epochs, el, coef, lla, ...
                                  sigma_c_m, label, ...
                                  sinexFile, stationName, isIGS)
% VTEC_MultiConstellation_calc  Batch VTEC inversion across GPS, Galileo, GLONASS, BeiDou.
%
% State vector layout (column order in design matrix G):
%   [VTEC_1, VTEC_2, ..., VTEC_N,   <-- one per surviving epoch  (cols 1..N)
%    dVTEC_lat,                      <-- single constant lat gradient (col N+1)
%    dVTEC_lon,                      <-- single constant lon gradient (col N+2)
%    Bias_1, Bias_2, ..., Bias_M]    <-- one per unique Measure type  (cols N+3 .. N+2+M)
%
% INPUTS
%   nPRNdata    - GPS     PRNdata struct array
%   lPRNdata    - Galileo PRNdata struct array
%   gPRNdata    - GLONASS PRNdata struct array
%   bPRNdata    - BeiDou  PRNdata struct array
%   nSat        - vector of GPS     PRN indices to include
%   lSat        - vector of Galileo PRN indices to include
%   gSat        - vector of GLONASS PRN indices to include
%   bSat        - vector of BeiDou  PRN indices to include
%   timeGNSS    - datetime/duration array of all candidate epoch times
%   epochs      - number of epochs to sample (evenly spaced from timeGNSS)
%   el          - elevation mask [degrees]
%   coef        - NeQuick ionospheric coefficients (3×1 from RINEX header)
%   lla         - receiver geodetic position [lat_deg, lon_deg, alt_m]
%   sigma_c_m   - NeQuick constraint tightness in equivalent metres (default 5.0 m).
%                 Conceptually: the pseudorange-equivalent uncertainty you assign to
%                 the NeQuick prediction. Larger → looser coupling to the model;
%                 smaller → tighter coupling (NeQuick dominates the solution).
%   label       - string label for figure titles (e.g. 'Multi-Constellation')
%   sinexFile   - path to a ground-station SINEX bias file.  Pass '' or [] to
%                 skip the truth comparison entirely.
%   stationName - 4-character IGS/GNSS station identifier (e.g. 'PARK').
%                 Case-insensitive; matched against the SINEX station column.
%   isIGS       - logical scalar. true → station is an IGS reference station
%                 (affects figure labelling only; no algorithmic difference).
%
% OUTPUTS
%   nPRNdataOUT  - GPS     PRNdata with .Bias field populated
%   lPRNdataOUT  - Galileo PRNdata with .Bias field populated
%   gPRNdataOUT  - GLONASS PRNdata with .Bias field populated
%   bPRNdataOUT  - BeiDou  PRNdata with .Bias field populated
%   TECANS        - full solution vector [VTEC; dVTEC_lat; dVTEC_lon; Biases]
%   Bias          - estimated biases for every unique Measure type (M×1) [m]
%   K_all         - epoch×satellite matrix of PRN bookkeeping (cell array, one per constellation)
%   DCBcomparison - struct with fields:
%                     .measureType  – cell array of qualified Measure strings
%                     .estimated    – estimated bias [m]
%                     .truth        – SINEX truth DCB [m]  (NaN if not found)
%                     .error        – estimated − truth    [m]
%                     .stationName  – station identifier string
%                     .isIGS        – logical flag

% =========================================================================
% 0. Input defaults
% =========================================================================
    if nargin < 14 || isempty(sigma_c_m)
        sigma_c_m = 5.0;   % [m] default: moderately soft NeQuick constraint
    end
    if nargin < 15 || isempty(label)
        label = 'Multi-Constellation';
    end
    if nargin < 16
        sinexFile = [];
    end
    if nargin < 17
        stationName = '';
    end
    if nargin < 18 || isempty(isIGS)
        isIGS = false;
    end
    useNeQuickConstraint = false;

    % Speed of light [m/ns] — needed to convert SINEX ns DCBs to metres
    c_mns = 299792458 * 1e-9;

    % Map qualified constellation name → single-char GNSS system identifier
    % (used when looking up truth DCBs in the SINEX struct)
    constSysChar = containers.Map( ...
        {'GPS','Galileo','GLONASS','BeiDou'}, ...
        {'G',  'E',      'R',      'C'     });

    % ---- Load ground-station truth DCBs from SINEX file (if provided) ----
    doTruthComparison = ~isempty(sinexFile) && ~isempty(stationName);
    stationBias = [];

    if doTruthComparison
        fprintf('Loading ground-station SINEX bias file: %s\n', sinexFile);
        try
            stationBias = groundStationBiasSinex(sinexFile);
            fprintf('  Loaded %d station entries.\n', numel(stationBias));
        catch ME
            warning('Could not load SINEX file (%s). Skipping truth comparison.', ME.message);
            doTruthComparison = false;
        end
    end

% =========================================================================
% 1. Bundle all four constellations into a single iterable struct
%    so the loops below can treat each constellation uniformly.
% =========================================================================
    consts(1).name    = 'GPS';
    consts(1).PRNdata = nPRNdata;
    consts(1).sat     = nSat;

    consts(2).name    = 'Galileo';
    consts(2).PRNdata = lPRNdata;
    consts(2).sat     = lSat;

    consts(3).name    = 'GLONASS';
    consts(3).PRNdata = gPRNdata;
    consts(3).sat     = gSat;

    consts(4).name    = 'BeiDou';
    consts(4).PRNdata = bPRNdata;
    consts(4).sat     = bSat;

    nConst = numel(consts);

% =========================================================================
% 2. Discover ALL unique Measure types across every constellation.
%    Each unique Measure type gets exactly one bias column.
%    E.g. {"GPS_L1L2", "GAL_L1L5", "GLO_L1L2", "BDS_L1L5"} → nBias = 4
%    The Measure strings already encode the signal combination; we prefix
%    with the constellation name to guarantee uniqueness even if two
%    constellations share the same raw Measure string.
% =========================================================================
    allMeasureLabels = {};
    for c = 1:nConst
        for j = 1:length(consts(c).sat)
            idx = consts(c).sat(j);
            rawMeas = consts(c).PRNdata(idx).Measure;
            if ~isempty(rawMeas)
                qualifiedMeas = sprintf('%s_%s', consts(c).name, rawMeas);
                allMeasureLabels{end+1} = qualifiedMeas; %#ok<AGROW>
            end
        end
    end
    measureTypes = unique(allMeasureLabels(:));   % M×1 cell array, sorted
    nBias = numel(measureTypes);
    % Count how many satellites use each measure type
    measSatCount = zeros(nBias, 1);
    for m = 1:nBias
        measSatCount(m) = sum(strcmp(allMeasureLabels, measureTypes{m}));
    end
    fprintf('Found %d unique Measure type(s) across all constellations:\n', nBias);
    for m = 1:nBias
        fprintf('  [%d] %s\n', m, measureTypes{m});
    end

% =========================================================================
% 3. Derive a reference beta (used only for the NeQuick scaling constant).
%    beta = (F1² - F2²) / (F1² · F2²)  [Hz⁻²]
%    S_neq = 1e16 · 40.3 · beta  converts VTEC in TECU to metres of
%    equivalent pseudorange-error at zenith (obliquity factor = 1).
%    We pick the first visible GPS satellite as the reference.
% =========================================================================
    ref_i    = consts(1).sat(1);
    F1_ref   = consts(1).PRNdata(ref_i).F1;
    F2_ref   = consts(1).PRNdata(ref_i).F2;
    beta_ref = (F1_ref^2 - F2_ref^2) / (F1_ref^2 * F2_ref^2);
    S_neq    = 1e16 * 40.3 * beta_ref;   % [m / TECU]

    % The constraint weight is:
    %   w_c = S_neq / sigma_c_m   [dimensionless]
    %
    % Physical interpretation:
    %   • Each real pseudorange observation row has an implicit weight of
    %     ~1/sigma_obs (where sigma_obs ≈ pseudorange noise in metres).
    %   • The NeQuick constraint row is designed to carry the same dimensional
    %     units. If we multiply both sides of "VTEC ≈ VTEC_NeQuick" by S_neq,
    %     the left side becomes an equivalent pseudorange-range (metres).
    %   • Dividing further by sigma_c_m normalises the row so that its
    %     contribution to the normal equations is equivalent to an observation
    %     with noise sigma_c_m metres — directly comparable to real
    %     pseudorange rows.
    %   • Result: sigma_c_m = 1 m makes NeQuick as tight as a 1-m pseudorange;
    %             sigma_c_m = 5 m is moderately soft;
    %             sigma_c_m = 50 m is nearly free (data-driven VTEC).
    constraint_weight = S_neq / sigma_c_m;   % [m/TECU] / [m] = [1/TECU]

% =========================================================================
% 4. Select epoch indices (evenly spaced) and count valid observations
%    per epoch across ALL constellations.
% =========================================================================
    epoch_indices = round(linspace(1, length(timeGNSS), epochs));

    % K_all{c}(k, j) = PRN index of the j-th satellite in constellation c
    %                   at epoch k, or NaN if not visible / invalid.
    K_all = cell(nConst, 1);
    for c = 1:nConst
        K_all{c} = NaN(length(timeGNSS), length(consts(c).sat));
    end

    % numSats_const(ep, c) = number of usable sats from constellation c at epoch ep
    numSats_const = NaN(length(epoch_indices), nConst);
    total_obs     = 0;   % accumulate total row count for pre-allocation
    removedCount  = 0;

    ep_idx = 0;
    epoch_indices_cell = num2cell(epoch_indices);   % working copy we may shrink

    idx_keep = true(size(epoch_indices));   % logical mask over epoch_indices

    for ep = 1:length(epoch_indices)
        k     = epoch_indices(ep);
        ep_idx = ep + 1;
        sats_this_epoch = 0;

        for c = 1:nConst
            l = 1;
            sat_c = 0;
            for j = 1:length(consts(c).sat)
                i = consts(c).sat(j);
                pd = consts(c).PRNdata(i);
                if ismember(timeGNSS(k), pd.PRNTT.Time) && ...
                   ~isnan(pd.PRNTT.PseudoDiff_cor(timeGNSS(k))) && ...
                   pd.PRNTT.el(timeGNSS(k)) >= el
                    K_all{c}(k, l) = i;
                    l = l + 1;
                    sat_c = sat_c + 1;
                    sats_this_epoch = sats_this_epoch + 1;
                end
            end
            numSats_const(ep, c) = sat_c;
        end

        % Minimum observations = unknowns + 1 for redundancy.
        % State vector has:  1 VTEC + 2 gradients (constant) + nBias  per epoch.
        % The 2 gradient unknowns are SHARED across all epochs so the per-epoch
        % minimum is 1 + nBias observatons, but to be safe we also require
        % at least (3 + nBias) per epoch (same criterion as original).
        min_obs_needed = 3 + nBias;
        if sats_this_epoch < min_obs_needed
            idx_keep(ep) = false;
            removedCount = removedCount + 1;
            K_all{1}(k,:) = NaN;
            K_all{2}(k,:) = NaN;
            K_all{3}(k,:) = NaN;
            K_all{4}(k,:) = NaN;
            fprintf('Epoch %d removed (only %d sats across all constellations, need %d)\n', ...
                    k, sats_this_epoch, min_obs_needed);
        else
            total_obs = total_obs + sats_this_epoch;
        end
    end

    epoch_indices = epoch_indices(idx_keep);
    numSats_const = numSats_const(idx_keep, :);
    epochs        = length(epoch_indices);
    % Determine UTC hour for each surviving epoch (drives per-hour gradient columns)
    epoch_hours = hour(timeGNSS(epoch_indices));   % N×1 integer UTC hours
    uniqueHours = unique(epoch_hours);             % sorted unique hours present
    nHours      = numel(uniqueHours);
    fprintf('Per-hour gradient estimation: %d UTC hour(s): [%s]\n', ...
            nHours, num2str(uniqueHours'));

    if epochs == 0
        error('VTEC_MultiConstellation_calc: No valid epochs remain after elevation/data screening.');
    end

    fprintf('\n%d epoch(s) removed. %d epochs remain with %d total observations.\n', ...
            removedCount, epochs, total_obs);

% =========================================================================
% 5. Compute NeQuick VTEC reference for each surviving epoch.
%    A virtual satellite is placed directly overhead at GPS orbital altitude
%    (20 200 km) so that calcModel integrates the full ionospheric column
%    above the receiver → pure VTEC (no obliquity factor needed).
% =========================================================================
    fprintf('Computing NeQuick VTEC reference for %d epochs...\n', epochs);
    NeQuick_VTEC  = zeros(epochs, 1);
    SatPos_zenith = [lla(1), lla(2), 202e5];   % directly overhead at GPS altitude

    % Replace per-epoch loop with a single vectorized call to calcModel.
    % Prepare UTC times for all surviving epochs (datevec expects an Nx6 array).
    UTC_times = datevec(timeGNSS(epoch_indices));   % epochs x 6

    try
        % Call calcModel once for all epochs. Assume calcModel accepts an
        % array of UTC rows and returns vectors of results in the same order.
        % First output (ignored) and TEC vector for each epoch returned in TEC_all.
        [~, TEC_all] = calcModel(UTC_times, repmat(lla, size(UTC_times,1), 1), repmat(SatPos_zenith, size(UTC_times,1), 1), 1, coef, "Above");
        % Ensure TEC_all is a column vector matching epochs
        NeQuick_VTEC(:) = TEC_all(:);
    catch ME
        % If a single-call fails, fall back to per-epoch calls for robustness,
        % logging the error and continuing with NaNs where calcModel fails.
        warning('Vectorized calcModel failed (%s). Falling back to per-epoch calls.', ME.message);
        for ep = 1:epochs
            k = epoch_indices(ep);
            UTC_k = datevec(timeGNSS(k));
            try
                [~, TEC_k] = calcModel(UTC_k, lla, SatPos_zenith, 1, coef, "Above");
                NeQuick_VTEC(ep) = TEC_k;
            catch ME2
                warning('NeQuick failed at epoch %d (%s). Substituting NaN.', k, ME2.message);
                NeQuick_VTEC(ep) = NaN;
            end
        end
    end

    % Fill any NaN NeQuick values with the session median to prevent
    % the constraint right-hand side from corrupting the solution.
    nq_median = median(NeQuick_VTEC, 'omitmissing');
    NeQuick_VTEC(isnan(NeQuick_VTEC)) = nq_median;

    fprintf('  NeQuick VTEC range: [%.2f, %.2f] TECU  (mean: %.2f TECU)\n', ...
            min(NeQuick_VTEC), max(NeQuick_VTEC), mean(NeQuick_VTEC));

% =========================================================================
% 6. Build the design matrix G and observation vector y.
%
%  STATE VECTOR x  (total length = N + 2 + M)
%  ─────────────────────────────────────────────────────────────────────
%  Cols  1 … N        : VTEC at each of the N surviving epochs  [TECU]
%  Col   N+1          : dVTEC/dLat  (single constant)          [TECU/deg]
%  Col   N+2          : dVTEC/dLon  (single constant)          [TECU/deg]
%  Cols  N+3 … N+2+M  : inter-frequency/inter-system biases    [m]
%
%  OBSERVATION MODEL  (row for satellite i at epoch ep)
%  ─────────────────────────────────────────────────────────────────────
%  PseudoDiff_cor = OF · 1e16 · 40.3 · β_i · VTEC_ep
%                 + OF · 1e16 · 40.3 · β_i · Δφ_i  · dVTEC_lat
%                 + OF · 1e16 · 40.3 · β_i · Δλ_i  · dVTEC_lon
%                 + Bias_{Measure(i)}
%  where:
%    OF    = obliquity factor for satellite i at this epoch
%    β_i   = (F1²−F2²)/(F1²·F2²) for satellite i's own frequencies
%    Δφ_i  = IPP latitude  offset from receiver [deg]
%    Δλ_i  = IPP longitude offset from receiver [deg]
% =========================================================================
    N        = epochs;
    stateLen = N + 2*nHours + nBias;
    % State layout: [VTEC_1..N | dLat_h1 dLon_h1 | dLat_h2 dLon_h2 | ... | Bias_1..M]

    G = zeros(total_obs, stateLen);   % design matrix
    y = zeros(total_obs, 1);          % observations

    row = 1;   % running row index into G and y

    for ep = 1:epochs
        k       = epoch_indices(ep);
        vtec_col = ep;   % column index for VTEC at this epoch

        for c = 1:nConst
            nSats_c = nnz(~isnan(K_all{c}(k,:)));
            for m = 1:nSats_c
                i  = K_all{c}(k, m);
                pd = consts(c).PRNdata(i);

                % Per-satellite dual-frequency scaling factor β_i
                beta_i = (pd.F1^2 - pd.F2^2) / (pd.F1^2 * pd.F2^2);

                % Common scale: OF · 1e16 · 40.3 · β_i  [m / TECU]
                OF_i   = pd.PRNTT.OF(timeGNSS(k));
                scale  = OF_i * 1e16 * 40.3 * beta_i;

                % --- VTEC column (epoch-specific) ---
                % Coefficient for VTEC_ep: converts VTEC in TECU to
                % equivalent pseudorange difference in metres.
                G(row, vtec_col) = scale;

                % --- Constant lat gradient column (N+1) ---
                % Δφ = IPP latitude minus receiver latitude [deg].
                % Multiplying by scale converts [TECU/deg]·[deg] → [m].
                ep_hour = hour(timeGNSS(k));
                hIdx    = find(uniqueHours == ep_hour, 1);   % index into uniqueHours
                G(row, N + 2*(hIdx-1) + 1) = scale * pd.PRNTT.Dphi(timeGNSS(k));
                G(row, N + 2*(hIdx-1) + 2) = scale * pd.PRNTT.Dlambda(timeGNSS(k));

                % --- Bias column (constellation-qualified Measure type) ---
                % One-hot encoding: places a 1 in the column corresponding
                % to this satellite's qualified Measure type.
                qualMeas = sprintf('%s_%s', consts(c).name, pd.Measure);
                mIdx     = find(strcmp(measureTypes, qualMeas), 1);
                if isempty(mIdx)
                    error('Measure "%s" not found in measureTypes — should not happen.', qualMeas);
                end
                G(row, N + 2*nHours + mIdx) = 1;   % bias column index: N+2*nHours+mIdx

                % Observation: carrier-smoothed differential pseudorange [m]
                y(row) = pd.PRNTT.PseudoDiff_cor(timeGNSS(k));

                row = row + 1;
            end
        end
    end

% =========================================================================
% 7. Append soft NeQuick constraint rows — one per surviving epoch.
%
%  The constraint enforces:
%     VTEC_ep ≈ NeQuick_VTEC(ep)
%
%  To make this row commensurate with the pseudorange observation rows
%  (which are in metres), we multiply both sides by S_neq [m/TECU]:
%     S_neq · VTEC_ep ≈ S_neq · NeQuick_VTEC(ep)
%
%  Then we normalise by sigma_c_m [m] to set the relative weight:
%     (S_neq / sigma_c_m) · VTEC_ep ≈ (S_neq / sigma_c_m) · NeQuick_VTEC(ep)
%
%  Effect:
%   • The Euclidean norm of this constraint row equals (S_neq/sigma_c_m),
%     which matches a pseudorange observation with noise ~ sigma_c_m metres.
%   • If sigma_c_m equals the typical pseudorange noise (~0.3–1 m), the
%     constraint is as strong as one pseudorange observation.
%   • If sigma_c_m = 5 m (default), the constraint is soft and the data
%     can deviate from NeQuick while still being gently anchored to it.
%   • The gradient and bias columns are left at zero for the constraint rows
%     because we are constraining only the VTEC, not the gradients or biases.
% =========================================================================
    if useNeQuickConstraint
        % Build soft NeQuick constraint rows — one per surviving epoch
        C_rows = zeros(N, stateLen);
        C_rhs  = zeros(N, 1);

        for ep = 1:N
            vtec_col             = ep;
            C_rows(ep, vtec_col) = constraint_weight;
            C_rhs(ep)            = constraint_weight * NeQuick_VTEC(ep);
        end

        G_aug = [G;      C_rows];
        y_aug = [y;      C_rhs ];
        fprintf('NeQuick constraint active  (sigma_c = %.2f m).\n', sigma_c_m);
    else
        % Unconstrained — solve from pseudorange observations alone.
        % Note: the system may be poorly conditioned without this constraint;
        % pinv will handle rank deficiency but VTEC and biases may trade off freely.
        G_aug = G;
        y_aug = y;
        fprintf('NeQuick constraint DISABLED — data-driven solution only.\n');
    end
% =========================================================================
% 8. Solve the augmented least-squares system via pseudo-inverse.
%
%  x = pinv(G_aug' · G_aug) · G_aug' · y_aug
%
%  Using pinv (rather than \ or inv) on the normal matrix is deliberate:
%   • Before the NeQuick constraints are added, the VTEC / bias null-space
%     makes the system rank-deficient.
%   • The constraint rows regularise the system, but numerical rank
%     deficiency can still arise (e.g. if a bias type has very few
%     observations).  pinv handles this gracefully by zeroing the
%     contribution of near-singular directions.
% =========================================================================
    NtN  = G_aug' * G_aug;
    Nty  = G_aug' * y_aug;
    TECANS = pinv(NtN) * Nty;

    % ---- Extract state sub-vectors ----
    VTEC         = TECANS(1:N);                               % [TECU] per epoch
    dVTEC_lat    = TECANS(N+1 : 2 : N+2*nHours-1);           % [TECU/deg] nHours×1
    dVTEC_lon    = TECANS(N+2 : 2 : N+2*nHours  );           % [TECU/deg] nHours×1
    Bias         = TECANS(N+2*nHours+1 : N+2*nHours+nBias);  % [m] per Measure type

    fprintf('\n--- Estimated biases ---\n');
    for m = 1:nBias
        fprintf('  %-25s : %+.4f m\n', measureTypes{m}, Bias(m));
    end
    for h = 1:nHours
        fprintf('  dVTEC/dLat  UTC%02d:00  : %+.6f TECU/deg\n', uniqueHours(h), dVTEC_lat(h));
        fprintf('  dVTEC/dLon  UTC%02d:00  : %+.6f TECU/deg\n', uniqueHours(h), dVTEC_lon(h));
    end

% =========================================================================
% 8.5  Compare estimated biases to SINEX ground-station truth DCBs.
%
%  For each estimated Measure type (e.g. "GPS_C1C-C1W") we:
%   1. Extract the constellation name and the raw obs-pair string.
%   2. Map the constellation to a single-char system code (G/E/R/C).
%   3. Search stationBias for an entry whose .system and .name match
%      the provided stationName, and whose .obs contains the raw obs pair.
%   4. Convert the matching SINEX offset from nanoseconds to metres
%      using c [m/ns], then compute the signed error: estimated − truth.
%
%  All results are stored in the DCBcomparison output struct.
% =========================================================================
    % Pre-fill DCBcomparison with NaN truths (populated below if SINEX is available)
    DCBcomparison.measureType = measureTypes;
    DCBcomparison.estimated   = Bias;
    DCBcomparison.truth       = NaN(nBias, 1);
    DCBcomparison.error       = NaN(nBias, 1);
    DCBcomparison.nSats       = measSatCount;
    DCBcomparison.stationName = stationName;
    DCBcomparison.isIGS       = isIGS;

    % Map from your Measure strings → the SINEX obs-pair string(s) they correspond to.
    % GPS Measure strings already ARE obs pairs, so they map to themselves.
    % Galileo/GLONASS/BeiDou use human-readable band names that must be translated.
    % Multiple candidates exist where two RINEX attribute codes map to the same band
    % (e.g. C1C/C5Q and C1X/C5X both produce Measure = 'E1 - E5a').
    % Map from your Measure strings → the SINEX obs-pair string(s) they correspond to.
    measToSinexObs = containers.Map('KeyType','char','ValueType','any');
    
    % GPS
    measToSinexObs('C1C-C5Q') = {'C1C-C5Q'};
    measToSinexObs('C1C-C5I') = {'C1C-C5I'}; % Label now matches the I-channel logic
    measToSinexObs('C1C-C2L') = {'C1C-C2L'};
    measToSinexObs('C1C-C2W') = {'C1C-C2W'};
    
    % Galileo
    measToSinexObs('E1 - E5a') = {'C1C-C5Q', 'C1X-C5X'};
    measToSinexObs('E1 - E5')  = {'C1C-C8Q', 'C1X-C8X'};
    measToSinexObs('E1 - E5b') = {'C1C-C7Q', 'C1X-C7X'};
    measToSinexObs('E1 - E6')  = {'C1C-C6C', 'C1X-C6X'};
    
    % GLONASS
    measToSinexObs('G1 - G3') = {'C1C-C3X', 'C1C-C3I', 'C1C-C3Q'}; % Expanded for 3.03 compliance
    measToSinexObs('G1 - G2') = {'C1C-C2C'};
    
    % BeiDou
    measToSinexObs('B1 - B2Q') = {'C2I-C7Q'};
    measToSinexObs('B1 - B2I') = {'C2I-C7I'};
    measToSinexObs('B1 - B2X') = {'C2I-C7X'}; % Replaced B2D to align with standard X channel
    measToSinexObs('B1 - B3I') = {'C2I-C6I'};
    measToSinexObs('L1 - B1')  = {'C1P-C2I'};
    measToSinexObs('B1 - B2a') = {'C2I-C5P'}; % Maintained: Valid if receiver exports RINEX 3.04+

    if doTruthComparison
        stationNameUpper = upper(strtrim(stationName));

        for m = 1:nBias
            % Split qualified Measure type into constellation + obs pair
            % measureTypes entries are like "GPS_C1C-C1W"
            % Split off the constellation prefix (e.g. "GPS_C1C-C2W" → "GPS" + "C1C-C2W")
            underIdx  = strfind(measureTypes{m}, '_');
            if isempty(underIdx)
                continue
            end
            constName = measureTypes{m}(1 : underIdx(1)-1);       % e.g. 'GPS'
            rawMeas   = measureTypes{m}(underIdx(1)+1 : end);      % e.g. 'C1C-C2W' or 'E1 - E5a'

            % Resolve rawMeas to the SINEX obs-pair candidate(s)
            if isKey(measToSinexObs, rawMeas)
                candidateObs = measToSinexObs(rawMeas);
            else
                % Unknown Measure string — try treating it as a direct obs pair
                candidateObs = {rawMeas};
                warning('Section 8.5: No SINEX obs-pair mapping found for Measure "%s". Trying as-is.', rawMeas);
            end

            if ~isKey(constSysChar, constName)
                continue
            end
            sysChar = constSysChar(constName);   % e.g. 'G'

            % Find the matching SINEX entry: system + station name
            match = [];
            for s = 1:numel(stationBias)
                if strcmp(stationBias(s).system, sysChar) && ...
                   strcmpi(strtrim(stationBias(s).name), stationNameUpper)
                    match = stationBias(s);
                    break
                end
            end

            if isempty(match)
                fprintf('  [DCB truth] No SINEX entry found for station %s / system %s\n', ...
                        stationNameUpper, sysChar);
                continue
            end

            % Search for any of the candidate obs pairs in the SINEX entry
            obsCell  = cellstr(match.obs);
            obsIdx   = [];
            matchedObs = '';
            for ob = 1:numel(candidateObs)
                idx = find(strcmp(obsCell, candidateObs{ob}), 1);
                if ~isempty(idx)
                    obsIdx     = idx;
                    matchedObs = candidateObs{ob};
                    break
                end
            end

            if isempty(obsIdx)
                fprintf('  [DCB truth] None of {%s} found in SINEX for %s/%s\n', ...
                        strjoin(candidateObs, ', '), stationNameUpper, sysChar);
                continue
            end

            % Convert SINEX ns → metres
            truthVal_m = c_mns * match.offset(obsIdx);
            % sign convention is opposite VTEC sign convention
            DCBcomparison.truth(m) = -truthVal_m;
            DCBcomparison.error(m) = Bias(m) + truthVal_m;
        end

        % ---- Print comparison table ----
        igsTag = '';
        if isIGS
            igsTag = ' [IGS]';
        end
        fprintf('  %-25s  %10s  %10s  %10s  %6s\n', ...
                'Measure', 'Est (m)', 'Truth (m)', 'Error (m)', 'N Sats');
        fprintf('  %s\n', repmat('-', 1, 70));
        for m = 1:nBias
            if isnan(DCBcomparison.truth(m))
                fprintf('  %-25s  %+10.4f  %10s  %10s  %6d\n', ...
                        measureTypes{m}, Bias(m), 'N/A', 'N/A', measSatCount(m));
            else
                fprintf('  %-25s  %+10.4f  %+10.4f  %+10.4f  %6d\n', ...
                        measureTypes{m}, Bias(m), ...
                        DCBcomparison.truth(m), DCBcomparison.error(m), measSatCount(m));
            end
        end

        % RMS error across all Measure types that had a truth value
        validErr = DCBcomparison.error(~isnan(DCBcomparison.error));
        if ~isempty(validErr)
            rmsErr = sqrt(mean(validErr.^2));
            fprintf('  %s\n', repmat('-', 1, 62));
            fprintf('  RMS error (matched entries): %.4f m\n', rmsErr);
        end
    end
% =========================================================================
% 8.6  Truth VTEC — re-solve with SINEX truth DCBs held fixed.
%   For each Measure type that has a matched SINEX truth, substitute it;
%   fall back to the estimated bias where truth is unavailable.
%   The bias columns are then removed from G and the contribution is
%   subtracted from y, leaving a cleaner solve for [VTEC; gradients].
% =========================================================================
VTEC_truth = [];   % remains empty when no truth data is available
if doTruthComparison && any(~isnan(DCBcomparison.truth))

    % Build full truth bias vector, falling back to estimated where NaN
    truth_biases_full = DCBcomparison.truth;
    noBias = isnan(truth_biases_full);
    truth_biases_full(noBias) = Bias(noBias);

    % Remove bias contribution from observations
    G_bias_cols      = G(:, N+2*nHours+1 : N+2*nHours+nBias);
    y_truth_debiased = y - G_bias_cols * truth_biases_full;

    % Reduced design matrix: VTEC epochs + per-hour gradient columns only
    G_truth = G(:, 1 : N+2*nHours);

    % Re-apply NeQuick constraint (VTEC columns only; gradient cols stay 0)
    if useNeQuickConstraint
        C_truth = zeros(N, N+2*nHours);
        for ep_t = 1:N
            C_truth(ep_t, ep_t) = constraint_weight;
        end
        G_truth_aug = [G_truth;          C_truth                        ];
        y_truth_aug = [y_truth_debiased; constraint_weight * NeQuick_VTEC];
    else
        G_truth_aug = G_truth;
        y_truth_aug = y_truth_debiased;
    end

    TECANS_truth = pinv(G_truth_aug' * G_truth_aug) * G_truth_aug' * y_truth_aug;
    VTEC_truth   = TECANS_truth(1:N);   % [TECU]
    fprintf('\nTruth-bias VTEC range: [%.2f, %.2f] TECU  (mean: %.2f TECU)\n', ...
            min(VTEC_truth), max(VTEC_truth), mean(VTEC_truth));
end
% =========================================================================
% 9. Write estimated biases back into each constellation's PRNdata struct.
% =========================================================================
useSINEXTruth = true;

    updated = struct('GPS',nPRNdata,'Galileo',lPRNdata,'GLONASS',gPRNdata,'BeiDou',bPRNdata);
    fieldNames = {'GPS','Galileo','GLONASS','BeiDou'};

    % Determine whether we should prefer SINEX truth biases:
    preferSINEX = useSINEXTruth && isIGS;

    for c = 1:nConst
        for j = 1:length(consts(c).sat)
            i        = consts(c).sat(j);
            qualMeas = sprintf('%s_%s', consts(c).name, consts(c).PRNdata(i).Measure);
            mIdx     = find(strcmp(measureTypes, qualMeas), 1);
            if ~isempty(mIdx)
                % By default use the estimated Bias
                assignedBias = Bias(mIdx);
                % If SINEX truth is preferred and we have a matching truth value,
                % use the SINEX truth instead (fall back to estimate when NaN).
                if preferSINEX && doTruthComparison && ~isempty(DCBcomparison) && isfield(DCBcomparison,'truth')
                    % Find the index in DCBcomparison that matches this measure type
                    % (measureTypes and DCBcomparison entries align by index mIdx)
                    sinexVal = DCBcomparison.truth(mIdx);
                    if ~isnan(sinexVal)
                        assignedBias = sinexVal;
                    end
                end
                consts(c).PRNdata(i).Bias = assignedBias;
            else
                warning('Could not assign bias: Measure "%s" not in measureTypes.', qualMeas);
            end
        end
        updated.(fieldNames{c}) = consts(c).PRNdata;
    end

    nPRNdataOUT = updated.GPS;
    lPRNdataOUT = updated.Galileo;
    gPRNdataOUT = updated.GLONASS;
    bPRNdataOUT = updated.BeiDou;

% =========================================================================
% 10. Plotting
% =========================================================================
    constColors = {'#0072BD','#D95319','#77AC30','#7E2F8E'};  % GPS/GAL/GLO/BDS
    t_ep = timeGNSS(epoch_indices);

    % Determine figure layout: add an extra subplot row when truth data exists
    hasTruth = doTruthComparison && any(~isnan(DCBcomparison.truth));
    if hasTruth
        nRows   = 4;
        figH    = 1150;
        spVTEC  = [1  2  3 ];
        spSats  = 4;
        spBias  = 5;
        spGrad  = 6;
        spDCB   = [7  8  9 ];
        spEl    = [10 11 12];
    else
        nRows   = 3;
        figH    = 900;
        spVTEC  = [1  2  3];
        spSats  = 4;
        spBias  = 5;
        spGrad  = 6;
        spEl    = [7  8  9];
    end
    nCols = 3;

    igsTag = '';
    if isIGS && ~isempty(stationName)
        igsTag = sprintf(' — %s [IGS]', upper(strtrim(stationName)));
    elseif ~isempty(stationName)
        igsTag = sprintf(' — %s', upper(strtrim(stationName)));
    end

    fig = figure('Name', sprintf('%s — VTEC Estimation', label), ...
                 'Position', [50 50 1400 figH]);

    % ---- 10a. VTEC vs NeQuick ----
    ax1 = subplot(nRows, nCols, spVTEC);
    hold(ax1, 'on');

    % NeQuick shaded band (±sigma_c_m converted back to TECU)
    nq_hi = NeQuick_VTEC + (sigma_c_m / S_neq);
    nq_lo = NeQuick_VTEC - (sigma_c_m / S_neq);
    valid = isfinite(t_ep) & isfinite(nq_hi) & isfinite(nq_lo);
    t_v   = t_ep(valid);
    hi_v  = nq_hi(valid);
    lo_v  = nq_lo(valid);
    if numel(t_v) > 1
        [t_v, si] = sort(t_v);
        hi_v = hi_v(si);
        lo_v = lo_v(si);
        patch(ax1, [t_v; flipud(t_v)], [hi_v; flipud(lo_v)], 'r', ...
              'FaceAlpha', 0.12, 'EdgeColor', 'none', ...
              'DisplayName', sprintf('NeQuick \\pm%.1f m equiv.', sigma_c_m));
    end

    plot(ax1, t_ep, NeQuick_VTEC, 'r--', 'LineWidth', 2, ...
         'DisplayName', sprintf('NeQuick (\\sigma_c = %.1f m)', sigma_c_m));
    plot(ax1, t_ep, VTEC, 'b.-', 'LineWidth', 1.5, 'MarkerSize', 8, ...
         'DisplayName', 'Estimated VTEC');
    if ~isempty(VTEC_truth)
        plot(ax1, t_ep, VTEC_truth, 'g.-', 'LineWidth', 1.5, 'MarkerSize', 8, ...
             'DisplayName', 'Truth-bias VTEC (SINEX)');
    end
    xlabel(ax1, 'Time'); ylabel(ax1, 'VTEC (TECU)');
    title(ax1, 'Vertical Total Electron Content — Multi-Constellation Estimate');
    legend(ax1, 'Location', 'best');
    grid(ax1, 'on');

    % Annotation box showing key parameters
    annotation('textbox', [0.13, 0.88, 0.22, 0.08], ...
        'String', {sprintf('El. cutoff : %g°', el), ...
                   sprintf('Epochs     : %d', epochs), ...
                   sprintf('\\sigma_c    : %.1f m', sigma_c_m)}, ...
        'FitBoxToText', 'on', 'BackgroundColor', 'white', 'EdgeColor', 'black', ...
        'FontSize', 8);

    % ---- 10b. Satellites in view per constellation ----
    ax2 = subplot(nRows, nCols, spSats);
    hold(ax2, 'on');
    constNames = {'GPS','Galileo','GLONASS','BeiDou'};
    for c = 1:nConst
        plot(ax2, t_ep, numSats_const(:, c), '.-', ...
             'Color', constColors{c}, 'LineWidth', 1.2, ...
             'DisplayName', constNames{c});
    end
    plot(ax2, t_ep, sum(numSats_const, 2), 'k-', 'LineWidth', 2, ...
         'DisplayName', 'Total');
    xlabel(ax2, 'Time'); ylabel(ax2, 'Satellites in View');
    title(ax2, 'Satellite Count per Constellation');
    legend(ax2, 'Location', 'best');
    grid(ax2, 'on');

    % ---- 10c. Bias bar chart (estimated only, or grouped estimated vs truth) ----
    ax5 = subplot(nRows, nCols, spBias);

    if hasTruth
        % Grouped bar: col 1 = estimated, col 2 = truth
        truthForPlot = DCBcomparison.truth;   % NaN entries stay NaN
        barData = [Bias, truthForPlot];
        barsH = bar(ax5, 1:nBias, barData, 'grouped');
        barsH(1).FaceColor = 'flat';
        barsH(2).FaceColor = [0.85 0.85 0.85];   % light grey for truth
        barsH(2).EdgeColor = [0.4  0.4  0.4];
        % Colour estimated bars by constellation
        for m = 1:nBias
            for c2 = 1:nConst
                if startsWith(measureTypes{m}, constNames{c2})
                    barsH(1).CData(m,:) = hex2rgb(constColors{c2});
                    break
                end
            end
        end
        legend(ax5, {'Estimated','SINEX Truth'}, 'Location', 'best');
    else
        barsH = bar(ax5, 1:nBias, Bias, 'FaceColor', 'flat');
        % Colour each bar by constellation
        for m = 1:nBias
            for c2 = 1:nConst
                if startsWith(measureTypes{m}, constNames{c2})
                    barsH.CData(m,:) = hex2rgb(constColors{c2});
                    break
                end
            end
        end
    end

    ax5.XTick      = 1:nBias;
    ax5.XTickLabel = strrep(measureTypes, '_', '\_');
    ax5.XTickLabelRotation = 30;
    ylabel(ax5, 'Bias (m)');
    title(ax5, 'Estimated Inter-Freq/Sys Biases');
    grid(ax5, 'on');
    yline(ax5, 0, 'k--', 'LineWidth', 1);
    % ---- 10d. Plot lat/lon gradients vs UTC hour (dVTEC parameters vs time) ----
    ax4 = subplot(nRows, nCols, spGrad);
    hold(ax4, 'on');

    % Prepare time vector for plotting: convert uniqueHours to datetimes on an arbitrary date
    % Use same timezone as t_ep if available; otherwise use today's date.
    try
        baseDate = dateshift(t_ep(1), 'start', 'day');
    catch
        baseDate = datetime('today');
    end
    t_hours = baseDate + hours(uniqueHours(:));

    % Plot latitude and longitude gradients with distinct styles
    p1 = plot(ax4, t_hours, dVTEC_lat(:), '-o', 'LineWidth', 1.5, 'MarkerSize', 6, ...
              'DisplayName', 'dVTEC / dLat (TECU/deg)');
    p2 = plot(ax4, t_hours, dVTEC_lon(:), '-s', 'LineWidth', 1.5, 'MarkerSize', 6, ...
              'DisplayName', 'dVTEC / dLon (TECU/deg)');

    % Formatting
    xlabel(ax4, 'UTC Time');
    ylabel(ax4, 'Gradient (TECU / deg)');
    title(ax4, sprintf('Spatial Gradients per UTC Hour  (N = %d hr)', nHours));
    legend(ax4, 'Location', 'best');
    grid(ax4, 'on');

    % Improve x-axis tick labels to show hours
    ax4.XTick = t_hours;
    ax4.XTickLabel = compose('%02d:00', hour(t_hours));
    datetickFormat = 'HH:MM';
    % Ensure x-limits include full range
    xlim(ax4, [t_hours(1) - minutes(30), t_hours(end) + minutes(30)]);
    hold(ax4, 'off');
    % ---- 10e. DCB error plot (only when truth data is available) ----
    if hasTruth
        ax3 = subplot(nRows, nCols, spDCB);
        hold(ax3, 'on');

        validMask = ~isnan(DCBcomparison.error);
        xAll      = 1:nBias;

        % Bar coloured by constellation, grey-out entries with no truth
        barsE = bar(ax3, xAll, DCBcomparison.error, 'FaceColor', 'flat');
        for m = 1:nBias
            if ~validMask(m)
                barsE.CData(m,:) = [0.75 0.75 0.75];   % grey = no truth
            else
                for c2 = 1:nConst
                    if startsWith(measureTypes{m}, constNames{c2})
                        barsE.CData(m,:) = hex2rgb(constColors{c2});
                        break
                    end
                end
            end
        end

        yline(ax3, 0, 'k--', 'LineWidth', 1.2);
        if any(validMask)
            rmsVal = sqrt(mean(DCBcomparison.error(validMask).^2));
            yline(ax3,  rmsVal, 'r:', 'LineWidth', 1.2, ...
                  'DisplayName', sprintf('RMS = %.4f m', rmsVal));
            yline(ax3, -rmsVal, 'r:', 'LineWidth', 1.2, 'HandleVisibility','off');
            legend(ax3, sprintf('RMS = %.4f m', rmsVal), 'Location', 'best');
        end

        ax3.XTick             = xAll;
        ax3.XTickLabel        = strrep(measureTypes, '_', '\_');
        ax3.XTickLabelRotation = 30;
        ylabel(ax3, 'Error (m)');
        title(ax3, sprintf('DCB Error: Estimated − SINEX Truth  |  %s%s', ...
                           upper(strtrim(stationName)), igsTag));
        grid(ax3, 'on');
    end

    % ---- 10f. Elevation angle profiles per constellation ----
    ax6 = subplot(nRows, nCols, spEl);
    hold(ax6, 'on');
    for c = 1:nConst
        PRNlist_c = rmmissing(unique(K_all{c}(:)));
        for i = PRNlist_c'
            t_sat  = consts(c).PRNdata(i).PRNTT.Time;
            el_sat = consts(c).PRNdata(i).PRNTT.el;
            plot(ax6, t_sat, el_sat, '-', ...
                 'Color', [hex2rgb(constColors{c}), 0.45], ...
                 'LineWidth', 0.8, 'HandleVisibility', 'off');
        end
    end
    % Legend proxies
    for c = 1:nConst
        plot(ax6, NaT, NaN, '-', 'Color', constColors{c}, ...
             'LineWidth', 2, 'DisplayName', constNames{c});
    end
    yline(ax6, el, 'k--', 'LineWidth', 1.2, 'DisplayName', sprintf('Cutoff (%g°)', el));
    xlabel(ax6, 'Time'); ylabel(ax6, 'Elevation (deg)');
    title(ax6, 'Satellite Elevation Profiles');
    legend(ax6, 'Location', 'best');
    grid(ax6, 'on');

    sgtitle(fig, sprintf('%s — Batch Multi-Constellation VTEC Inversion%s', label, igsTag), ...
            'FontSize', 13, 'FontWeight', 'bold');

    % ---- Save figure ----
    try
        figName = sprintf('VTEC_MultiConst_%s_%s.png', ...
                          regexprep(label,'[^a-zA-Z0-9]','_'), ...
                          datestr(now, 'yyyy-mm-dd_HH-MM-SS'));
        saveas(gcf, figName);
        fprintf('Figure saved: %s\n', figName);
    catch ME
        warning('Could not save figure: %s', ME.message);
    end

end   % ── end VTEC_MultiConstellation_calc ──────────────────────────────


% =========================================================================
% Local helper: convert CSS hex colour string to [r g b] in [0,1]
% =========================================================================
function rgb = hex2rgb(hexStr)
    hexStr = strrep(hexStr, '#', '');
    rgb = double(reshape(sscanf(hexStr, '%2x'), 3, 1))' / 255;
end