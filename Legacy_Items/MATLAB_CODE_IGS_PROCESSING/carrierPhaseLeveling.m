function [PRNdataOUT, sat] = carrierPhaseLeveling(PRNdataIN, sat)
% carrierPhaseLeveling  Per-pass carrier-phase leveling for TEC estimation.
%
%   [PRNdataOUT] = carrierPhaseLeveling(PRNdataIN, sat)
%
%   Implements the carrier-phase leveling approach described in
%   Eq. (31.40) of "Ionospheric Effects, Monitoring, and Mitigation
%   Techniques". A per-pass leveling constant is computed as a
%   weighted mean of (pseudorange difference - carrier phase difference),
%   then applied to the carrier phase to produce a carrier-smoothed
%   pseudorange difference free of integer ambiguity bias.
%
%   Passes are defined as contiguous data segments with no time gap
%   exceeding GAP_THRESHOLD (default 10 minutes). Each pass must meet
%   a minimum duration of MIN_PASS_DURATION (default 20 minutes) for
%   the leveling to be considered reliable, per the document guidance.
%
%   INPUTS
%   ------
%   PRNdataIN : struct array  (same format as PseudoDiffCalc)
%       PRNdataIN(i).F1        - L1 frequency [Hz]
%       PRNdataIN(i).F2        - L2 frequency [Hz]
%       PRNdataIN(i).PRNTT     - MATLAB timetable with per-epoch data:
%           .CF1, .CF2         - L1/L2 pseudoranges [m]
%           .LF1, .LF2         - L1/L2 carrier-phase observables [cycles]
%           .D_adr  (optional) - pre-formed carrier-phase difference [m]
%           .el (optional) - satellite elevation angle [degrees],
%                                   used for measurement weighting
%   sat : vector of PRN indices to process
%
%   OUTPUTS
%   -------
%   PRNdataOUT : struct array (same as PRNdataIN, with added fields)
%       .PRNTT.PseudoDiff      - raw pseudorange difference Δρ₂₁ = CF2-CF1
%       .PRNTT.LevelingConst   - per-epoch leveling constant ΔNλ [m]
%       .PRNTT.PassID          - integer pass index for each epoch
%       .PRNTT.PseudoDiff_cor  - carrier-smoothed pseudorange difference
%                                (= CarrierDiff + LevelingConst) [m]
%       .PRNTT.TECu            - slant TEC [TECu]
%
%   ALGORITHM
%   ---------
%   For each pass of K epochs (Eq. 31.40):
%
%       ΔNλ₂₁ = Σ[ (1/σ²_k) · (Δρ₂₁_k - Δϕ₂₁_k) ]
%                ─────────────────────────────────────
%                        Σ[ 1/σ²_k ]
%
%   Weights 1/σ²_k follow a standard elevation-dependent model:
%       σ²_k = σ₀² / sin²(el_k)   if elevation is available
%       σ²_k = 1                   otherwise (equal weights)
%
%   The carrier-smoothed pseudorange difference is then:
%       Δρ_smoothed = Δϕ₂₁ + ΔNλ₂₁
%
%   NOTE: Carrier cycle slips should be detected and repaired in
%   PRNdataIN before calling this function. Passes shorter than
%   MIN_PASS_DURATION are leveled but flagged with a warning.

% ── Constants ────────────────────────────────────────────────────────────
c                = 299792458;   % speed of light [m/s]
GAP_THRESHOLD    = 10;          % minimum gap that starts a new pass [min]
MIN_PASS_DURATION = 20;         % minimum recommended pass length [min]

PRNdata = PRNdataIN;

for j = 1:length(sat)
    i = sat(j);

    % ── Frequencies & TEC scale factor ───────────────────────────────────
    F1    = PRNdata(i).F1;
    F2    = PRNdata(i).F2;
    BetaI = 1/40.3 * (F1^2 * F2^2) / (F1^2 - F2^2) * 1e-16;  % [TECu/m]

    TT = PRNdata(i).PRNTT;
    nEpochs = height(TT);
    if nEpochs == 0
        warning('PRN %d: no epochs available — removing from processing list.', i);
        % remove current satellite from sat vector so outer loop won't revisit it
        sat(j) = [];
        % adjust loop index to account for removed element
        j = j - 1;
        continue
    end

    % ── Raw pseudorange difference  Δρ₂₁ = ρ₂ - ρ₁ ──────────────────────
    PseudoDiff = TT.CF2 - TT.CF1;                              % [m]

    % ── Carrier-phase difference  Δϕ₂₁ ──────────────────────────────────
    if ismember('D_adr', TT.Properties.VariableNames) && ~isempty(TT.D_adr)
        fprintf('PRN %d: using D_adr for carrier-phase difference\n', i);
        CarrierDiff = TT.D_adr;
    else
        fprintf('PRN %d: computing carrier-phase difference from LF1/LF2\n', i);
        CarrierDiff = (c/F1) .* TT.LF1 - (c/F2) .* TT.LF2;   % [m]
    end

    % ── Elevation-based measurement weights  1/σ²_k ──────────────────────
    hasElev = ismember('el', TT.Properties.VariableNames) && ...
              ~all(isnan(TT.el));
    if hasElev
        el_rad = (TT.el);
        % σ² ∝ 1/sin²(el) → weight w = sin²(el); floor at 5° to avoid blow-up
        w = sin(max(el_rad, deg2rad(5))).^2;
    else
        w = ones(nEpochs, 1);
    % If any epochs produce a non-NaN corrected pseudorange, keep satellite.
    % Otherwise remove this PRN from future processing.
    % (Check after computing CarrierDiff and weights but before pass detection/leveling.)
    % Note: Carrier-smoothed pseudorange will be CarrierDiff + LevelingConst.
    % Since LevelingConst not yet computed, we can check if CarrierDiff is all NaN
    % and PseudoDiff is all NaN — then no valid measurements exist to produce PseudoDiff_cor.
    if all(isnan(PseudoDiff)) && all(isnan(CarrierDiff))
        warning('PRN %d: no valid measurements (pseudo and carrier) — removing from processing list.', i);
        sat(j) = [];
        j = j - 1;
        continue
    end
    end

    % ── Detect passes (time gaps > GAP_THRESHOLD minutes) ────────────────
    rowTimes  = TT.Properties.RowTimes;                        % datetime vector
    dt_min    = [Inf; minutes(diff(rowTimes))];                % gap before each epoch [min]
    passBreaks = find(dt_min > GAP_THRESHOLD);                 % indices that START a new pass
    passStart  = [passBreaks];
    passEnd    = [passBreaks(2:end) - 1; nEpochs];
    nPasses    = length(passStart);

    
    fprintf('PRN %d: %d pass(es) detected\n', i, nPasses);

    % ── Preallocate per-epoch output fields ───────────────────────────────
    LevelingConst = NaN(nEpochs, 1);   % ΔNλ₂₁ replicated for every epoch [m]
    PassID        = zeros(nEpochs, 1); % integer pass label

    % ── Per-pass leveling (Eq. 31.40) ────────────────────────────────────
    for p = 1:nPasses
        idx = passStart(p):passEnd(p);          % epoch indices for this pass

        % Duration check
        passDur_min = minutes(rowTimes(passEnd(p)) - rowTimes(passStart(p)));
        if passDur_min < MIN_PASS_DURATION
            warning(['PRN %d, pass %d: duration %.1f min < %d min minimum. ' ...
                     'Leveling may be unreliable.'], i, p, passDur_min, MIN_PASS_DURATION);
        end

        % Mask valid (non-NaN) epochs within this pass
        DeltaCMC = PseudoDiff(idx) - CarrierDiff(idx);    % Δρ₂₁ - Δϕ₂₁  [m]
        wPass    = w(idx);
        valid    = ~isnan(DeltaCMC) & ~isnan(wPass);

        if sum(valid) < 2
            warning('PRN %d, pass %d: insufficient valid epochs – skipping.', i, p);
            PassID(idx) = p;
            continue
        end

        % Weighted mean leveling constant  (Eq. 31.40)
        wv       = wPass(valid);
        DeltaCMCv = DeltaCMC(valid);
        Delta_Nlambda = sum(wv .* DeltaCMCv) / sum(wv);   % scalar [m]

        fprintf('  Pass %d | epochs: %d | duration: %.1f min | ΔNλ = %.4f m\n', ...
                p, sum(valid), passDur_min, Delta_Nlambda);

        % Broadcast the scalar leveling constant across all pass epochs
        LevelingConst(idx) = Delta_Nlambda;
        PassID(idx)        = p;
    end

    % ── Apply leveling: carrier-smoothed pseudorange difference ──────────
    %   Δρ_smoothed = Δϕ₂₁ + ΔNλ₂₁
    PseudoDiff_cor = CarrierDiff + LevelingConst;

    % ── Slant TEC ─────────────────────────────────────────────────────────
    TECu = BetaI .* PseudoDiff_cor;

    % ── Write results back into the timetable ────────────────────────────
    TT.PseudoDiff    = PseudoDiff;
    TT.LevelingConst = LevelingConst;
    TT.PassID        = PassID;
    TT.PseudoDiff_cor = PseudoDiff_cor;
    TT.TECu          = TECu;

    PRNdata(i).PRNTT = TT;
end

PRNdataOUT = PRNdata;
end