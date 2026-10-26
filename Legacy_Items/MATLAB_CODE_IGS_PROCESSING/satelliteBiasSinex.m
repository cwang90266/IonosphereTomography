function varargout = satelliteBiasSinex(filename, varargin)
    % Read and parse satellite bias data from SINEX format file
    % 
    % Inputs:
    %   filename - Path to SINEX bias file
    %   varargin - Variable number of PRN data structures (one per GNSS system)
    %              Expected order: GPS, GLONASS, Galileo, BeiDou
    %
    % Outputs:
    %   varargout - Updated PRN data structures with satellite bias information
    % by: Austin Hunter
    
    % Open the SINEX file for reading
    fid = fopen(filename);
    
    % Speed of light conversion factor: meters to nanoseconds
    c = 299792458 * 1e-9; % m/ns 
    
    % --- NEW: Measurement to SINEX Observable Mapping ---
    measToSinexObs = containers.Map('KeyType', 'char', 'ValueType', 'any');
    
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
    
    % Initialize SatBias fields for each satellite in each GNSS system
    for k = 1:length(varargin)
        for i = 1:length(varargin{k})
            varargin{k}(i).SatBias.obs = [];      % Observable type (e.g., 'C1C-C2W')
            varargin{k}(i).SatBias.offset = [];   % Bias offset value in nanoseconds
        end
    end
    
    % Skip lines until we reach the DSB (Differential Signal Bias) section
    current_line = fgetl(fid);
    while ~startsWith(current_line, ' DSB')
        current_line = fgetl(fid);
    end
    
    % Parse each DSB entry
    while startsWith(current_line, ' DSB')
        % Extract satellite system identifier (G=GPS, R=GLONASS, E=Galileo, C=BeiDou)
        system = current_line(12);
        
        % Extract satellite PRN number
        sat = str2double(current_line(13:14));
        
        % Extract first observable code (e.g., 'C1C')
        OBS1 = current_line(26:28);
        
        % Extract second observable code (e.g., 'C2W')
        OBS2 = current_line(31:33);
        
        % Extract bias value in nanoseconds
        Value = current_line(71:91);
        
        % Construct measurement identifier (e.g., 'C1C-C2W')
        meas = string(strcat(OBS1, '-', OBS2));
        
        % Map GNSS system to input argument index
        switch system
            case 'G' % GPS
                idx = 1;
            case 'R' % GLONASS
                idx = 2;
            case 'E' % Galileo
                idx = 3;
            case 'C' % BeiDou
                idx = 4;
            otherwise
                idx = [];
        end
        
        % Store bias data if valid system and satellite number
        if ~isempty(idx) && idx <= length(varargin) && ~isnan(sat)
            % Append observable type to list
            varargin{idx}(sat).SatBias.obs = [varargin{idx}(sat).SatBias.obs; meas];
            
            % Append bias offset value to list
            varargin{idx}(sat).SatBias.offset = [varargin{idx}(sat).SatBias.offset; str2double(Value)];
            
            % --- NEW: Mapped Comparison Logic ---
            % If this bias matches the mapped measurement type, apply the correction
            if isfield(varargin{idx}(sat), 'Measure')
                % Convert receiver measure to standard character array for map lookup
                rx_meas = char(varargin{idx}(sat).Measure);
                
                % Check if the receiver's measurement string exists in our dictionary
                if isKey(measToSinexObs, rx_meas)
                    % Retrieve the cell array of valid SINEX strings for this measurement
                    valid_sinex_obs = measToSinexObs(rx_meas);
                    
                    % Check if the current SINEX observable matches any in the valid list
                    if any(strcmp(valid_sinex_obs, char(meas)))
                        % Safety check: Only apply if PRNTT exists for this satellite
                        if isfield(varargin{idx}(sat), 'PRNTT')
                            varargin{idx}(sat).PRNTT.PseudoDiff_cor = ...
                                varargin{idx}(sat).PRNTT.PseudoDiff_cor + (c * str2double(Value));
                        end
                    end
                end
            end
        end
        
        % Check for end of bias section
        if contains(current_line, 'ENDBIA')
            break;
        end
        
        % Read next line
        current_line = fgetl(fid);
    end
    
    % Close the file
    fclose(fid);
    
    % Assign updated PRN data structures to output arguments
    for k = 1:nargout
        varargout{k} = varargin{k};
    end
end