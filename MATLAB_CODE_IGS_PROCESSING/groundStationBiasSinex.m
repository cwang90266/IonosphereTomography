function stationBias = groundStationBiasSinex(filename)
    % Read and parse ground station DCB data from a SINEX bias file
    %
    % Input:
    %   filename  - Path to SINEX bias file containing receiver/station DSB entries
    %
    % Output:
    %   stationBias - Struct array with one element per unique station, containing:
    %                   .name   - 4-character station ID (e.g., 'PARK')
    %                   .system - GNSS system char (G, R, E, C)
    %                   .obs    - String array of observable pairs (e.g., "C1C-C1W")
    %                   .offset - Double array of bias offsets in nanoseconds
    %
    % by: Austin Hunter

    fid = fopen(filename, 'r');
    if fid == -1
        error('Could not open file: %s', filename);
    end

    % Use a containers.Map to accumulate data keyed by "SYSTEM_STATION" (e.g., "G_PARK")
    biasMap = containers.Map('KeyType', 'char', 'ValueType', 'any');

    % Advance to the DSB section
    current_line = fgetl(fid);
    while ischar(current_line) && ~startsWith(current_line, ' DSB')
        current_line = fgetl(fid);
    end

    % Parse every DSB line in the station-bias section
    while ischar(current_line) && startsWith(current_line, ' DSB')

        % Bail out at end-of-section marker
        if contains(current_line, 'ENDBIA')
            break;
        end

        % Ground-station lines have a blank PRN field and a station name.
        % Format (1-based column indices):
        %   col  7      : GNSS system (G / R / E / C)
        %   col  12     : GNSS system repeated
        %   cols 16-19  : Station name (4 chars)
        %   cols 26-28  : OBS1 (e.g., C1C)
        %   cols 31-33  : OBS2 (e.g., C1W)
        %   cols 70-80  : Bias value in nanoseconds (right-justified)

        % Skip lines that are too short to be valid
        if length(current_line) < 91
            current_line = fgetl(fid);
            continue;
        end

        system  = current_line(7);
        station = strtrim(current_line(16:19));
        OBS1    = strtrim(current_line(26:28));
        OBS2    = strtrim(current_line(31:33));
        value   = str2double(strtrim(current_line(70:91)));  % widen window for station files

        % Skip entries with no station name (satellite-only lines) or bad values
        if isempty(station) || isnan(value)
            current_line = fgetl(fid);
            continue;
        end

        % Only handle known GNSS systems
        if ~ismember(system, {'G','R','E','C'})
            current_line = fgetl(fid);
            continue;
        end

        meas = string(strcat(OBS1, '-', OBS2));
        key  = sprintf('%s_%s', system, station);

        if isKey(biasMap, key)
            entry        = biasMap(key);
            entry.obs    = [entry.obs;    meas ];
            entry.offset = [entry.offset; value];
            biasMap(key) = entry;
        else
            entry.name   = station;
            entry.system = system;
            entry.obs    = meas;
            entry.offset = value;
            biasMap(key) = entry;
        end

        current_line = fgetl(fid);
    end

    fclose(fid);

    % Convert map to a plain struct array, sorted by system then station name
    keys         = biasMap.keys();
    sortedKeys   = sort(keys);
    nStations    = numel(sortedKeys);

    if nStations == 0
        stationBias = struct('name',{},'system',{},'obs',{},'offset',{});
        return;
    end

    stationBias(nStations) = struct('name','','system','','obs',[],'offset',[]);
    for k = 1:nStations
        stationBias(k) = biasMap(sortedKeys{k});
    end
end