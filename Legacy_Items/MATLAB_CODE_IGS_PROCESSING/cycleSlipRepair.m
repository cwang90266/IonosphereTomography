function PRNdataOUT = cycleSlipRepair(PRNdataIN, sat, sigmaBounds)
% CYCLESLIPREPAIR  Detects and corrects cycle slips in L1-L2 ADR carrier phase data.
%
%   PRNdataOUT = cycleSlipRepair(PRNdataIN, sat, sigmaBounds)
%
%   Inputs:
%     PRNdataIN   - Struct array of per-PRN data (fields: F1, F2, PRNTT)
%     sat         - Vector of PRN indices to process
%     sigmaBounds - Number of sigma for cycle-slip detection threshold
%
%   Output:
%     PRNdataOUT  - Updated struct with corrected LF2, D_adr, D_adr_dot,
%                   cycle_slip_detected, and SlipIndices fields populated.

    c       = 299792458;    % Speed of light (m/s)
    PRNdata = PRNdataIN;
    figOn   = false;

    % Alignment method: 'global_poly' | 'sequential_linear'
    alignment_method = 'sequential_linear';
    

    % -----------------------------------------------------------------
    for j = 1:length(sat)
        i = sat(j);
        % if i ~= 27
        %     continue
        % end

        fprintf("Detecting cycle slips on PRN %d\n", i);

        % --- Carrier phase difference: L1 - L2 (metres) ---
        L1_ADR = (c / PRNdata(i).F1) .* PRNdata(i).PRNTT.LF1;
        L2_ADR = (c / PRNdata(i).F2) .* PRNdata(i).PRNTT.LF2;
        D_adr  = L1_ADR - L2_ADR;
        Time   = PRNdata(i).PRNTT.Time;
        N      = length(D_adr);

        % Median sample interval (seconds) — used throughout for window sizing
        dt_vec    = seconds(diff(Time));
        dt_median = median(dt_vec, 'omitmissing');
        if dt_median == 30
            min_pts = 1;
        else
            min_pts          = 10;
        end
        % =================================================================
        % STEP 1 — IDENTIFY INITIAL DISCONTINUITY BOUNDARIES
        % =================================================================

        % (a) Large time gaps (> 3x median interval)
        disc_time = find((dt_vec > 2 * dt_median))+1;


        % (b) Long NaN runs (>= 5 consecutive); mark the END index
        nan_d      = diff([0; isnan(D_adr); 0]);
        nan_starts = find(nan_d ==  1);
        nan_ends   = find(nan_d == -1) - 1;
        if dt_median == 30
            long_runs  = (nan_ends - nan_starts + 1) >= 1;
        else
            long_runs  = (nan_ends - nan_starts + 1) >= 5;
        end
        disc_nan   = nan_ends(long_runs);

        discontinuity_indices = unique([disc_time(:); disc_nan(:)]);
        fprintf("  Initial discontinuities found: %d\n", length(discontinuity_indices));

        % =================================================================
        % STEP 2 — DYNAMIC OUTLIER & STRUCTURAL JUMP DETECTION
        % =================================================================
        segment_boundaries = unique([1; discontinuity_indices(:); N]);

        n = 1;
        while n < length(segment_boundaries)
            seg_start = segment_boundaries(n);

            % Last boundary is inclusive; all others define half-open intervals
            if n < length(segment_boundaries) - 1
                seg_end = segment_boundaries(n+1) - 1;
            else
                seg_end = segment_boundaries(n+1);
            end

            if seg_end <= seg_start
                n = n + 1;
                continue;
            end

            check_adr = D_adr(seg_start:seg_end);

            % Point-to-point differences (zero-padded to preserve length)
            adr_diff    = [0; diff(check_adr)];
            window_size = min(30, length(check_adr));

            if window_size >= 3
                local_med = movmedian(adr_diff, window_size, 'omitmissing');
                local_mad = movmedian(abs(adr_diff - local_med), window_size, 'omitmissing');

                % Replace zero MAD with robust global fallback to avoid infinite scaling
                zero_mad = (local_mad == 0);
                if any(zero_mad)
                    g_std = std(adr_diff, 'omitmissing') / 1.4826;
                    if g_std == 0 || isnan(g_std); g_std = 1e-3; end
                    local_mad(zero_mad) = g_std;
                end

                outlier_mask = abs(adr_diff - local_med) > 10 * local_mad;
            else
                outlier_mask = false(size(check_adr));
            end

            if ~any(outlier_mask)
                n = n + 1;
                continue;
            end

            if sum(outlier_mask) <= min_pts
                % Isolated spikes — NaN them in place, keep segment intact
                D_adr(seg_start + find(outlier_mask) - 1) = NaN;
                fprintf('  Segment %d: Removed %d isolated spike(s).\n', n, sum(outlier_mask));
                n = n + 1;
            else
                % Structural jump — locate the single largest step
                [~, max_diff_idx] = max(abs(diff(check_adr)));
                jump_local_idx    = max_diff_idx + 1;
                jump_global_idx   = seg_start + jump_local_idx - 1;
                pts_before        = jump_local_idx - 1;
                pts_remaining     = seg_end - jump_global_idx + 1;

                if pts_before <= min_pts
                    % Jump almost immediately — discard leading stub
                    D_adr(seg_start:jump_global_idx-1) = NaN;
                    fprintf('  Segment %d: Jump near start. Truncated first %d pts.\n', n, pts_before);
                    n = n + 1;

                elseif pts_remaining <= min_pts
                    % Jump at the very end — discard trailing stub
                    D_adr(jump_global_idx:seg_end) = NaN;
                    fprintf('  Segment %d: Jump near end. Truncated last %d pts.\n', n, pts_remaining);
                    n = n + 1;

                else
                    % Jump in the middle — split the segment
                    fprintf('  Segment %d: Split at index %d.\n', n, jump_global_idx);
                    segment_boundaries = [ segment_boundaries(1:n);   ...
                                           jump_global_idx;            ...
                                           segment_boundaries(n+1:end) ];

                 end
            end
        end % while

        % =================================================================
        % STEP 3 — PRUNE SMALL SEGMENTS
        % =================================================================
        remove_mask = false(N, 1);
        seg_lens   = diff(segment_boundaries);

        small_mask = seg_lens < 10;

       if any(small_mask)
            small_idxs    = find(small_mask);
            orig_n_bounds = length(segment_boundaries);   % snapshot BEFORE any deletions

            % --- Pass 1: precompute all data ranges from the unmodified boundary array ---
            nan_ranges = zeros(length(small_idxs), 2);
            for k = 1:length(small_idxs)
                s = small_idxs(k);
                nan_ranges(k, 1) = segment_boundaries(s);
                % Last boundary is inclusive; all others are half-open (exclusive end)
                if s < orig_n_bounds - 1
                    nan_ranges(k, 2) = segment_boundaries(s+1) - 1;
                else
                    nan_ranges(k, 2) = segment_boundaries(s+1);
                end
            end

           % --- Pass 2: NaN out all small segment data before touching boundaries ---
            for k = 1:length(small_idxs)
                s = small_idxs(k);
                fprintf('  Removing small segment %d (len %d < %d, idx %d-%d).\n', ...
                        s, seg_lens(s), min_pts, nan_ranges(k,1), nan_ranges(k,2));
                D_adr(nan_ranges(k,1):nan_ranges(k,2)) = NaN;
                remove_mask(nan_ranges(k,1):nan_ranges(k,2)) = true;
            end

            % --- Pass 3: collapse boundaries in reverse so earlier indices stay valid ---
            for k = length(small_idxs):-1:1
                s = small_idxs(k);
                segment_boundaries(s+1) = [];
            end
        end

        num_segments = length(segment_boundaries) - 1;
        seg_lens     = diff(segment_boundaries);
        [~, longest_seg_idx] = max(seg_lens);

        ref_start = segment_boundaries(longest_seg_idx);
        ref_end   = segment_boundaries(longest_seg_idx + 1) - 1;

        fprintf("  Final segments: %d  |  Reference: seg %d (idx %d-%d, len %d)\n", ...
                num_segments, longest_seg_idx, ref_start, ref_end, ref_end - ref_start + 1);
        
       D_adr_raw = D_adr;
        % =================================================================
        % STEP 4 — PHASE ALIGNMENT
        % =================================================================
        switch alignment_method

            % -------------------------------------------------------------
            case 'global_poly'
            % -------------------------------------------------------------
                fprintf("  Method: Global 2nd-Order Polynomial Alignment\n");

                % Trim NaN boundaries from reference before fitting
                ref_valid = trimNaNEnds(ref_start:ref_end, D_adr);
                if isempty(ref_valid)
                    warning('PRN %d: Reference segment has no valid data. Skipping.', i);
                    continue
                end

                t_ref = seconds(Time(ref_valid) - Time(ref_valid(1)));
                a_ref = D_adr(ref_valid);

                % Robust inlier selection via MAD
                med_r = median(a_ref, 'omitmissing');
                mad_r = median(abs(a_ref - med_r), 'omitmissing');
                sig_r = max(1.4826 * mad_r, std(a_ref, 'omitmissing'));
                if sig_r == 0 || isnan(sig_r)
                    in_mask = true(size(a_ref));
                else
                    in_mask = abs(a_ref - med_r) < 3 * sig_r;
                    if sum(in_mask) < max(3, round(0.5 * numel(a_ref)))
                        in_mask = abs(a_ref - med_r) < 5 * sig_r;
                    end
                end

                t_clean = t_ref(in_mask);
                a_clean = a_ref(in_mask);

                if numel(t_clean) >= 3
                    p_ref = polyfit(t_clean, a_clean, 2);
                elseif numel(t_clean) >= 2
                    fprintf("  Warning: Only %d clean pts — using 1st-order fit.\n", numel(t_clean));
                    p_ref = [0, polyfit(t_clean, a_clean, 1)];
                else
                    error('PRN %d: Reference segment has too few valid points to fit.', i);
                end

                % Align every non-reference segment to the global polynomial
                for seg = 1:num_segments
                    if seg == longest_seg_idx; continue; end

                    raw_idx   = segment_boundaries(seg):segment_boundaries(seg+1)-1;
                    seg_valid = trimNaNEnds(raw_idx, D_adr);
                    if isempty(seg_valid); continue; end

                    t_anch   = seconds(Time(seg_valid(1)) - Time(ref_valid(1)));
                    expected = polyval(p_ref, t_anch);
                    offset   = D_adr(seg_valid(1)) - expected;

                    if ~isfinite(offset)
                        fprintf("  Warning: Invalid offset for seg %d — skipping.\n", seg);
                        continue
                    end

                    D_adr(raw_idx) = D_adr(raw_idx) - offset;
                    fprintf("  Aligned seg %d,  offset = %.3f m\n", seg, offset);
                end

            % -------------------------------------------------------------
            case 'sequential_linear'
            % -------------------------------------------------------------
                fprintf("  Method: Sequential Linear Outward Alignment\n");

                % Boundary window: max(min_pts, samples in 2 minutes)
                win_2min = round(120 / dt_median);
                win_size = max(min_pts, win_2min);
                fprintf("  Boundary window: %d pts  (2-min ~= %d pts)\n", win_size, win_2min);

                % --- Forward pass: longest segment -> end of data ---
                for seg = (longest_seg_idx + 1):num_segments
                    prev_raw = segment_boundaries(seg-1):segment_boundaries(seg)-1;
                    curr_raw = segment_boundaries(seg):segment_boundaries(seg+1)-1;

                    prev_valid = trimNaNEnds(prev_raw, D_adr);
                    curr_valid = trimNaNEnds(curr_raw, D_adr);

                    % If either side is empty after trimming, walk backward to find
                    % the nearest previous segment that has valid data (for prev_valid)
                    % or the nearest next segment that has valid data (for curr_valid).
                    if isempty(prev_valid) || isempty(curr_valid)
                        % Find nearest previous segment with valid data for prev_valid
                        if isempty(prev_valid)
                            k = seg-1;
                            while k >= 1
                                cand_raw = segment_boundaries(k):segment_boundaries(k+1)-1;
                                cand_valid = trimNaNEnds(cand_raw, D_adr);
                                if ~isempty(cand_valid)
                                    prev_valid = cand_valid;
                                    break
                                end
                                k = k - 1;
                            end
                        end

                        % Find nearest next segment with valid data for curr_valid
                        if isempty(curr_valid)
                            k = seg;
                            while k <= num_segments
                                cand_raw = segment_boundaries(k):segment_boundaries(k+1)-1;
                                cand_valid = trimNaNEnds(cand_raw, D_adr);
                                if ~isempty(cand_valid)
                                    curr_valid = cand_valid;
                                    break
                                end
                                k = k + 1;
                            end
                        end

                        if isempty(prev_valid) || isempty(curr_valid)
                            fprintf('  Forward: Skipping seg %d (no nearby valid segment after NaN trim).\n', seg);
                            continue
                        end
                    end

                    % Localised window: TAIL of the previous (already-aligned) segment
                    win_idx = prev_valid( max(1, end - win_size + 1) : end );
                    p       = fitPoly1InWindow(Time, D_adr, win_idx);

                    % Extrapolate forward to the first valid epoch of the current segment
                    t_extrap = seconds(Time(curr_valid(1)) - Time(win_idx(1)));
                    expected = polyval(p, t_extrap);
                    offset   = D_adr(curr_valid(1)) - expected;

                    if ~isfinite(offset)
                        fprintf('  Forward: Invalid offset for seg %d — skipping.\n', seg);
                        continue
                    end

                    D_adr(curr_raw) = D_adr(curr_raw) - offset;
                    fprintf("  Forward: seg %d -> seg %d,  offset = %.3f m\n", seg, seg-1, offset);
                end

                % --- Backward pass: longest segment -> start of data ---
                for seg = (longest_seg_idx - 1):-1:1
                    next_raw = segment_boundaries(seg+1):segment_boundaries(seg+2)-1;
                    curr_raw = segment_boundaries(seg):segment_boundaries(seg+1)-1;

                    next_valid = trimNaNEnds(next_raw, D_adr);
                    curr_valid = trimNaNEnds(curr_raw, D_adr);

                    % If either side is empty after trimming, walk backward to find
                    % the nearest previous segment that has valid data (for prev_valid)
                    % or the nearest next segment that has valid data (for curr_valid).
                    if isempty(next_valid) || isempty(curr_valid)
                        % Find nearest previous segment with valid data for prev_valid
                        if isempty(next_valid)
                            k = seg+1;
                            while k <= num_segments
                                cand_raw = segment_boundaries(k):segment_boundaries(k+1)-1;
                                cand_valid = trimNaNEnds(cand_raw, D_adr);
                                if ~isempty(cand_valid)
                                    next_valid = cand_valid;
                                    break
                                end
                                k = k + 1;
                            end
                        end

                        % Find nearest next segment with valid data for curr_valid
                        if isempty(curr_valid)
                            k = seg;
                            while k <= num_segments
                                cand_raw = segment_boundaries(k):segment_boundaries(k+1)-1;
                                cand_valid = trimNaNEnds(cand_raw, D_adr);
                                if ~isempty(cand_valid)
                                    curr_valid = cand_valid;
                                    break
                                end
                                k = k + 1;
                            end
                        end

                        if isempty(next_valid) || isempty(curr_valid)
                            fprintf('  Forward: Skipping seg %d (no nearby valid segment after NaN trim).\n', seg);
                            continue
                        end
                    end

                    % Localised window: HEAD of the next (already-aligned) segment
                    win_idx = next_valid( 1 : min(end, win_size) );
                    p       = fitPoly1InWindow(Time, D_adr, win_idx);

                    % Extrapolate backward to the last valid epoch of the current segment
                    t_extrap = seconds(Time(curr_valid(end)) - Time(win_idx(1)));
                    expected = polyval(p, t_extrap);
                    offset   = D_adr(curr_valid(end)) - expected;

                    if ~isfinite(offset)
                        fprintf('  Backward: Invalid offset for seg %d — skipping.\n', seg);
                        continue
                    end

                    D_adr(curr_raw) = D_adr(curr_raw) - offset;
                    fprintf("  Backward: seg %d -> seg %d,  offset = %.3f m\n", seg, seg+1, offset);
                end

            otherwise
                error('Invalid alignment_method: ''%s''.', alignment_method);
        end % switch

        fprintf("  Discontinuity correction complete.\n");

       

        % =================================================================
        % STEP 6 — CYCLE SLIP DETECTION ON CORRECTED D_ADR
        % =================================================================
        D_adr_dot = [NaN; diff(D_adr) ./ dt_vec];

        % Robust global statistics — pre-filter extreme outliers
        valid_dots = D_adr_dot(isfinite(D_adr_dot));
        med_dot    = median(valid_dots);
        rough_std  = std(valid_dots);
        clean_dots = valid_dots(abs(valid_dots - med_dot) < 5 * rough_std);

        mu_global    = mean(clean_dots);
        sigma_global = std(clean_dots);

        upper_bound = mu_global + sigmaBounds * sigma_global;
        lower_bound = mu_global - sigmaBounds * sigma_global;

        cycle_slips        = (D_adr_dot > upper_bound) | (D_adr_dot < lower_bound);
        cycle_slip_indices = find(cycle_slips);

        % Store pre-correction snapshot and detection masks
        PRNdata(i).PRNTT.D_adr              = D_adr;
        PRNdata(i).PRNTT.D_adr_dot          = D_adr_dot;
        PRNdata(i).PRNTT.cycle_slip_detected = cycle_slips;
        PRNdata(i).SlipIndices              = cycle_slip_indices;

        % Remove any remaining jump discontinuities from D_adr
        for s = 1:length(cycle_slip_indices)
            idx = cycle_slip_indices(s);
            if idx > 1 && isfinite(D_adr(idx)) && isfinite(D_adr(idx-1))
                jump           = D_adr(idx) - D_adr(idx-1);
                D_adr(idx:end) = D_adr(idx:end) - jump;
            end
        end

        % % Reconstruct L2 carrier phase: D_adr = L1_ADR - L2_ADR
        % PRNdata(i).PRNTT.LF2 = (L1_ADR - D_adr) * PRNdata(i).F2 / c;

        fprintf("  Found %d cycle slips.\n", length(cycle_slip_indices));

        
        % =========================================================================
        % COMBINED PLOT: ALIGNMENT & CYCLE SLIP OVERVIEW
        % =========================================================================
        if figOn
            % Create a large figure to comfortably fit 6 subplots
            figure('Name', sprintf('PRN %d: Processing Overview', i), ...
                   'NumberTitle', 'off');%, 'Position', [100, 100, 1600, 900]);
            
            % ---------------------------------------------------------------------
            % LEFT COLUMN: SEGMENTS & ALIGNMENT
            % ---------------------------------------------------------------------
            
            % --- Subplot 1: All Segments (Raw) ---
            subplot(3,2,1); 
            hold on; grid on;
            colors_seg = lines(max(1,num_segments));
            for s = 1:num_segments
                idx = segment_boundaries(s):segment_boundaries(s+1)-1;
                valid = ~isnan(D_adr(idx));
                if ~any(valid)
                    continue
                end
                plot(Time(idx(valid)), D_adr_raw(idx(valid)), '.-', ...
                     'Color', colors_seg(mod(s-1,size(colors_seg,1))+1,:), ...
                     'DisplayName', sprintf('Seg %d (len %d)', s, sum(valid)));
                % mark segment boundaries
                xline(Time(idx(1)), '--', 'Color', colors_seg(mod(s-1,size(colors_seg,1))+1,:), 'LineWidth',0.5);
                xline(Time(idx(end)), '--', 'Color', colors_seg(mod(s-1,size(colors_seg,1))+1,:), 'LineWidth',0.5);
            end
            % highlight reference segment
            ref_x = [Time(ref_start), Time(ref_end)];
            ylimits = ylim;
            patch([ref_x(1) ref_x(2) ref_x(2) ref_x(1)], [ylimits(1) ylimits(1) ylimits(2) ylimits(2)], ...
                  [0.9 0.9 0.9], 'FaceAlpha', 0.15, 'EdgeColor','none', 'DisplayName','Reference Segment');
            title('1. Segment Overview (Raw)');
            ylabel('Delta ADR (m)');
            legend('Location','best', 'NumColumns', 2);
            
            % --- Pre-compute Alignment Polynomial (Shared by Subplots 3 & 5) ---
            valid_mask = isfinite(D_adr);
            time_full  = seconds(Time - Time(ref_start));
            if strcmp(alignment_method, 'sequential_linear')
                p_vis = polyfit(time_full(valid_mask), D_adr(valid_mask), 2);
            else
                p_vis = p_ref; % Already computed above
            end
            D_adr_fitted = polyval(p_vis, time_full);
            colors_align = lines(num_segments);
        
            % --- Subplot 3: Aligned segments ---
            subplot(3,2,3); 
            hold on; grid on;
            for seg = 1:num_segments
                si  = segment_boundaries(seg):segment_boundaries(seg+1)-1;
                lbl = sprintf('Segment %d', seg);
                if seg == longest_seg_idx
                    plot(Time(si), D_adr(si), 'o-', 'Color', colors_align(seg,:), ...
                         'LineWidth', 2, 'MarkerSize', 4, 'DisplayName', [lbl ' (Ref)']);
                else
                    plot(Time(si), D_adr(si), '.-', 'Color', colors_align(seg,:), ...
                         'LineWidth', 1, 'MarkerSize', 3, 'DisplayName', lbl);
                end
            end
            plot(Time, D_adr_fitted, 'k--', 'LineWidth', 2, 'DisplayName', 'Global Poly (2nd-order)');
            for d = 1:length(discontinuity_indices)
                xline(Time(min(discontinuity_indices(d)+1, N)), 'r:', 'LineWidth', 1.5, 'HandleVisibility','off');
            end
            ylabel('L1-L2 ADR (m)');
            title(sprintf('3. Segments Aligned (%s)', strrep(alignment_method,'_',' ')));
            legend('Location', 'best', 'NumColumns', 2);
        
            % --- Subplot 5: Residuals ---
            subplot(3,2,5); 
            hold on; grid on;
            residuals = D_adr - D_adr_fitted;
            for seg = 1:num_segments
                si = segment_boundaries(seg):segment_boundaries(seg+1)-1;
                if seg == longest_seg_idx
                    plot(Time(si), residuals(si), 'o-', 'Color', colors_align(seg,:), 'LineWidth', 2, 'MarkerSize', 4);
                else
                    plot(Time(si), residuals(si), '.-', 'Color', colors_align(seg,:), 'LineWidth', 1, 'MarkerSize', 3);
                end
            end
            for d = 1:length(discontinuity_indices)
                xline(Time(min(discontinuity_indices(d)+1, N)), 'r:', 'LineWidth', 1.5);
            end
            yline(0, 'k--', 'LineWidth', 1);
            ylabel('Residual (m)'); xlabel('Time');
            title('5. Residuals from Global Trend'); 
        
            % ---------------------------------------------------------------------
            % RIGHT COLUMN: CYCLE SLIPS
            % ---------------------------------------------------------------------
        
            % --- Subplot 2: Rate of change + detection bounds ---
            subplot(3,2,2); 
            hold on; grid on;
            plot(Time, D_adr_dot, 'b-', 'LineWidth', 1, 'DisplayName', '$\Delta\dot{\mathrm{ADR}}$');
            yline(upper_bound, 'r--', 'LineWidth', 1.5, 'Label', sprintf('+%.0f sigma', sigmaBounds), 'DisplayName', 'Upper Bound');
            yline(lower_bound, 'r--', 'LineWidth', 1.5, 'Label', sprintf('-%.0f sigma', sigmaBounds), 'DisplayName', 'Lower Bound');
            yline(mu_global, 'k:', 'LineWidth', 1, 'Label', 'Mean', 'DisplayName', 'Mean');
            if ~isempty(cycle_slip_indices)
                plot(Time(cycle_slip_indices), D_adr_dot(cycle_slip_indices), ...
                     'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'DisplayName', 'Cycle Slips');
            end
            ylabel('d(L1-L2 ADR)/dt  (m/s)');
            title('2. Cycle Slip Detection - Rate of Change');
            legend('Location', 'best', 'Interpreter', 'latex');
        
            % --- Subplot 4: D_adr before slip correction ---
            subplot(3,2,4); 
            hold on; grid on;
            plot(Time, PRNdata(i).PRNTT.D_adr, 'b.', 'LineWidth', 1);
            for s = 1:length(cycle_slip_indices)
                xline(Time(cycle_slip_indices(s)), 'r--', 'LineWidth', 1.5);
            end
            ylabel('L1-L2 ADR (m)');
            title('4. L1-L2 ADR - Pre Slip Correction');
        
            % --- Subplot 6: D_adr after slip correction ---
            subplot(3,2,6); 
            hold on; grid on;
            plot(Time, D_adr, 'g.', 'LineWidth', 1, 'DisplayName', 'Corrected');
            ylabel('L1-L2 ADR (m)'); xlabel('Time');
            title('6. L1-L2 ADR - Post Slip Correction');
            legend('Location', 'best'); 
        
            % --- Add a Master Title ---
            sgtitle(sprintf('PRN %d: Processing Overview (Alignment & Cycle Slips)', i), ...
                    'FontSize', 16, 'FontWeight', 'bold');
            saveas(gcf,strcat('figures\',sprintf('cycleSlipRepair_PRN_%d.png',i)),'png')
        end
        PRNdata(i).PRNTT.D_adr              = D_adr;
        PRNdata(i).PRNTT.D_adr_dot          = D_adr_dot;
        PRNdata(i).PRNTT.cycle_slip_detected = cycle_slips;
        PRNdata(i).SlipIndices              = cycle_slip_indices;

        % =================================================================
        % STEP 7 — PERMANENTLY REMOVE SHORT SEGMENTS FROM PRNTT
        % =================================================================
        if any(remove_mask)
            fprintf("  Purging %d short-segment indices from PRNTT.\n", sum(remove_mask));
            
            % 1. Map SlipIndices to the new shorter array indexing
            keep_mask = ~remove_mask;
            new_idx_map = cumsum(keep_mask);
            PRNdata(i).SlipIndices = new_idx_map(PRNdata(i).SlipIndices);
            
            % 2. Remove the rows/elements from the PRNTT data structure
            if istable(PRNdata(i).PRNTT) || istimetable(PRNdata(i).PRNTT)
                PRNdata(i).PRNTT(remove_mask, :) = [];
            elseif isstruct(PRNdata(i).PRNTT)
                fields = fieldnames(PRNdata(i).PRNTT);
                for f = 1:numel(fields)
                    % Only remove from fields that match the full array length
                    if numel(PRNdata(i).PRNTT.(fields{f})) == N
                        PRNdata(i).PRNTT.(fields{f})(remove_mask) = [];
                    end
                end
            end
        end
    end % for j
    
    PRNdataOUT = PRNdata;
end % cycleSlipRepair


% =========================================================================
%  LOCAL HELPER FUNCTIONS
% =========================================================================

function valid_idx = trimNaNEnds(indices, data)
% TRIMMNANENDS  Strip leading/trailing NaN or Inf positions from an index range.
%
%   valid_idx = trimNaNEnds(indices, data)
%
%   Returns the sub-range of INDICES such that DATA(valid_idx(1)) and
%   DATA(valid_idx(end)) are both finite.  Internal NaNs are preserved so
%   they can be excluded individually inside polyfit via the 'valid' mask.

    if isempty(indices)
        valid_idx = [];
        return
    end

    finite_mask = isfinite(data(indices));

    if ~any(finite_mask)
        valid_idx = [];
        return
    end

    first_v   = find(finite_mask, 1, 'first');
    last_v    = find(finite_mask, 1, 'last');
    valid_idx = indices(first_v:last_v);
end


function p = fitPoly1InWindow(Time, D_adr, win_idx)
% FITPOLY1INWINDOW  Robust 1st-order polynomial fit over a localised window.
%
%   p = fitPoly1InWindow(Time, D_adr, win_idx)
%
%   The time axis is anchored to WIN_IDX(1) so that evaluation at a query
%   epoch uses the same origin:
%
%       t_query = seconds(Time(query_idx) - Time(win_idx(1)));
%       y_hat   = polyval(p, t_query);
%
%   Gracefully degrades to a flat line when fewer than 2 valid points exist.

    if isempty(win_idx)
        p = [0, 0];
        return
    end

    t_win   = seconds(Time(win_idx) - Time(win_idx(1)));   % relative time (s)
    adr_win = D_adr(win_idx);
    valid   = isfinite(t_win) & isfinite(adr_win);

    n_valid = sum(valid);
    if n_valid >= 2
        p = polyfit(t_win(valid), adr_win(valid), 1);
    elseif n_valid == 1
        p = [0, adr_win(valid)];   % flat-line through the single available point
    else
        p = [0, 0];                % degenerate fallback
    end
end