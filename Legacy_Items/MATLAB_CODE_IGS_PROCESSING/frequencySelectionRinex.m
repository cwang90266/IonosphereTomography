function [PRNdata, sat,time] = frequencySelectionRinex(data,system, initialtime, endtime)
% Function: Takes Rinexread data and does a frequency selection protocol to
% choose the frequencies for dual-frequency analysis. 
% By Austin Hunter
% Created: 2024.06.11

%% Inputs
% data = data of a 1 x 1 struct from the rinexread function on RINEX observation data
% system = choice of GNSS system: GPS, Galileo, GLONASS, BeiDou
% initialtime = character array in DD_Month_YYY HH:MM:SS form
% endtime = character array in DD_Month_YYY HH:MM:SS form

%% Outputs
% PRNdata = output of RINEX data seperated by PRN for one constellation
% sat = array of satellite PRNs present in the observation data
% time = datetime array of all time indicies in the observation data

% Code eliminates all data columns with greater than:
p = 0.2;
% missing data

%% Frequency Definitions

GPSL1 = 1575.42e6;
GPSL2 = 1227.6e6;
GPSL5 = 1176.45e6;

GALE1 = 1575.42e6;
GALE6 = 1278.75e6;
GALE5a = 1176.45e6;
GALE5b = 1207.14e6;
GALE5 = 1191.795e6;

% Why are the GLONASS frequencies a range?
GLOL1 = 1598.0625e6; %1609.3125e6;
GLOL2 = 1242.9375e6; %1251.6875e6;
GLOL3 = 1202.025e6;

BDSB1 = 1561.098e6;
BDSB2a = 1176.45e6;
BDSB2 = 1207.14e6;
BDSB3 = 1268.520e6;



if system == "GPS"
        %% GPS
        fprintf("GPS\n")
        time = unique(data.GPS.Time);
        [~,f] =  min(abs(time - initialtime));
        [~,e] =  min(abs(time - endtime));
        time = time(f:e);
        [~,first] = min(abs(data.GPS.Time - initialtime));
        [~,last] = min(abs(data.GPS.Time - endtime));
        
        [~,I] = sort(data.GPS.SatelliteID(first:last));
        GPSdata = data.GPS(first:last,:);
        GPSdata = GPSdata(I,:);
        %%
        for i = 1:32
        idx = find(~(GPSdata.SatelliteID(:)-i));
        PRNTT = GPSdata(idx,:);
        PRNdata(i) = Var_Names(PRNTT);
        n = floor(p*height(PRNTT));
        PRNdata(i).PRNTT = rmmissing(PRNdata(i).PRNTT,2,'MinNumMIssing',n);
        end
        
        sat = unique(data.GPS.SatelliteID(first:last));
        
        %% Establish heirarchy of GPS signals to use for TEC Estimation
        % array to remove bad TEC calculation sattelites from satGPS
        REMOVE =[];
        
        for j = 1:length(sat)
            i = sat(j);
            
            VarNames = string(PRNdata(i).PRNTT.Properties.VariableNames);
        
        if any(strcmp(VarNames,"C1C")) && any(strcmp(VarNames,"C5Q")) && any(strcmp(VarNames,"L1C")) && any(strcmp(VarNames,"L5Q"))
            PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["C1C","C5Q","L1C","L5Q"],["CF1","CF2","LF1","LF2"]);
            PRNdata(i).F1 = GPSL1;
            PRNdata(i).F2 = GPSL5;
            PRNdata(i).Measure = "C1C-C5Q";
            fprintf('PRN %i will calculate TEC with %s data \n',i,PRNdata(i).Measure)
            if any(strcmp(VarNames,"S1C")) && any(strcmp(VarNames,"S5Q"))
                PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["S1C","S5Q"],["SF1","SF2"]);
            end
        elseif  any(strcmp(VarNames,"C1C")) && any(strcmp(VarNames,"C5I")) && any(strcmp(VarNames,"L1C")) && any(strcmp(VarNames,"L5I"))
            PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["C1C","C5I","L1C","L5I"],["CF1","CF2","LF1","LF2"]);
            PRNdata(i).F1 = GPSL1;
            PRNdata(i).F2 = GPSL5;
            PRNdata(i).Measure = "C1C-C5I";
            fprintf('PRN %i will calculate TEC with %s data \n',i,PRNdata(i).Measure)
            if any(strcmp(VarNames,"S1C")) && any(strcmp(VarNames,"S5I"))
                PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["S1C","S5I"],["SF1","SF2"]);
            end
        % elseif any(strcmp(VarNames,"C1C")) && any(strcmp(VarNames,"C2L")) && any(strcmp(VarNames,"L1C")) && any(strcmp(VarNames,"L2L"))
        %     PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["C1C","C2L","L1C","L2L"],["CF1","CF2","LF1","LF2"]);
        %     PRNdata(i).F1 = GPSL1;
        %     PRNdata(i).F2 = GPSL2;
        %     PRNdata(i).Measure = "C1C-C2L";
        %     fprintf('PRN %i will calculate TEC with %s data \n',i,PRNdata(i).Measure)
        %     if any(strcmp(VarNames,"S1C")) && any(strcmp(VarNames,"S2L"))
        %         PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["S1C","S2L"],["SF1","SF2"]);
        %     end
        elseif any(strcmp(VarNames,"C1C")) && any(strcmp(VarNames,"C2W")) && any(strcmp(VarNames,"L1C")) && any(strcmp(VarNames,"L2W"))
            PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["C1C","C2W","L1C","L2W"],["CF1","CF2","LF1","LF2"]);
            PRNdata(i).F1 = GPSL1;
            PRNdata(i).F2 = GPSL2;
            PRNdata(i).Measure = 'C1C-C2W';
            fprintf('PRN %i will calculate TEC with %s data \n',i,PRNdata(i).Measure)
            if any(strcmp(VarNames,"S1C")) && any(strcmp(VarNames,"S2W"))
                PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["S1C","S2W"],["SF1","SF2"]);
            end
        else
            fprintf('PRN %i does not have sufficient dual-frequency data \n',i)
            REMOVE = [REMOVE,i];
        end
        
        end
        
        %%
        if ~isempty(REMOVE)
        sat = sat(all(sat ~= REMOVE,2));
        end
        %Remove this
        % sat = sat(sat ~=16);
elseif system == "Galileo"
        %% Galileo
        fprintf("Galileo\n")
        time = unique(data.Galileo.Time);
        [~,f] =  min(abs(time - initialtime));
        [~,e] =  min(abs(time - endtime));
        time = time(f:e);
        [~,first] = min(abs(data.Galileo.Time - initialtime));
        [~,last] = min(abs(data.Galileo.Time - endtime));
        [~,I] = sort(data.Galileo.SatelliteID(first:last));
        GALdata = data.Galileo(first:last,:);
        GALdata = GALdata(I,:);
        %%
for i = 1:36
        idx = find(~(GALdata.SatelliteID(:)-i));
        PRNTT = GALdata(idx,:);
        PRNdata(i) = Var_Names(PRNTT);
        n = floor(p*height(PRNTT));
        PRNdata(i).PRNTT = rmmissing(PRNdata(i).PRNTT,2,'MinNumMIssing',n);
end
        sat = unique(data.Galileo.SatelliteID(first:last));
        %% Establish heirarchy of GAL signals to use for TEC Estimation
        % array to remove bad TEC calculation sattelites from satGPS
            REMOVE =[];
        for j = 1:length(sat)
                    i = sat(j);
                    VarNames = string(PRNdata(i).PRNTT.Properties.VariableNames);
        if  any(strcmp(VarNames,"C1C")) && any(strcmp(VarNames,"C5Q")) && any(strcmp(VarNames,"L1C")) && any(strcmp(VarNames,"L5Q"))
                    PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["C1C","C5Q","L1C","L5Q"],["CF1","CF2","LF1","LF2"]);
                    PRNdata(i).F1 = GALE1;
                    PRNdata(i).F2 = GALE5a;
                    PRNdata(i).Measure = 'E1 - E5a';
                    fprintf('PRN E%i will calculate TEC with %s data \n',i,PRNdata(i).Measure)
        if any(strcmp(VarNames,"S1C")) && any(strcmp(VarNames,"S5Q"))
                        PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["S1C","S5Q"],["SF1","SF2"]);
        end
        elseif any(strcmp(VarNames,"C1X")) && any(strcmp(VarNames,"C5X")) && any(strcmp(VarNames,"L1X")) && any(strcmp(VarNames,"L5X"))
                    PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["C1X","C5X","L1X","L5X"],["CF1","CF2","LF1","LF2"]);
                    PRNdata(i).F1 = GALE1;
                    PRNdata(i).F2 = GALE5a;
                    PRNdata(i).Measure = 'E1 - E5a';
                    fprintf('PRN E%i will calculate TEC with %s data \n',i,PRNdata(i).Measure)
        if any(strcmp(VarNames,"S1X")) && any(strcmp(VarNames,"S5X"))
                        PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["S1X","S5X"],["SF1","SF2"]);
        end
        elseif any(strcmp(VarNames,"C1C")) && any(strcmp(VarNames,"C8Q")) && any(strcmp(VarNames,"L1C")) && any(strcmp(VarNames,"L8Q"))
                    PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["C1C","C8Q","L1C","L8Q"],["CF1","CF2","LF1","LF2"]);
                    PRNdata(i).F1 = GALE1;
                    PRNdata(i).F2 = GALE5;
                    PRNdata(i).Measure = 'E1 - E5';
                    fprintf('PRN E%i will calculate TEC with %s data \n',i,PRNdata(i).Measure)
        if any(strcmp(VarNames,"S1C")) && any(strcmp(VarNames,"S8Q"))
                        PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["S1C","S8Q"],["SF1","SF2"]);
        end
        elseif any(strcmp(VarNames,"C1X")) && any(strcmp(VarNames,"C8X")) && any(strcmp(VarNames,"L1X")) && any(strcmp(VarNames,"L8X"))
                    PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["C1X","C8X","L1X","L8X"],["CF1","CF2","LF1","LF2"]);
                    PRNdata(i).F1 = GALE1;
                    PRNdata(i).F2 = GALE5;
                    PRNdata(i).Measure = 'E1 - E5';
                    fprintf('PRN E%i will calculate TEC with %s data \n',i,PRNdata(i).Measure)
        if any(strcmp(VarNames,"S1X")) && any(strcmp(VarNames,"S8X"))
                        PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["S1X","S8X"],["SF1","SF2"]);
        end
        elseif any(strcmp(VarNames,"C1C")) && any(strcmp(VarNames,"C7Q")) && any(strcmp(VarNames,"L1C")) && any(strcmp(VarNames,"L7Q"))
                    PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["C1C","C7Q","L1C","L7Q"],["CF1","CF2","LF1","LF2"]);
                    PRNdata(i).F1 = GALE1;
                    PRNdata(i).F2 = GALE5b;
                    PRNdata(i).Measure = 'E1 - E5b';
                    fprintf('PRN E%i will calculate TEC with %s data \n',i,PRNdata(i).Measure)
        if any(strcmp(VarNames,"S1C")) && any(strcmp(VarNames,"S7Q"))
                        PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["S1C","S7Q"],["SF1","SF2"]);
        end
        elseif any(strcmp(VarNames,"C1X")) && any(strcmp(VarNames,"C7X")) && any(strcmp(VarNames,"L1X")) && any(strcmp(VarNames,"L7X"))
                    PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["C1X","C7X","L1X","L7X"],["CF1","CF2","LF1","LF2"]);
                    PRNdata(i).F1 = GALE1;
                    PRNdata(i).F2 = GALE5b;
                    PRNdata(i).Measure = 'E1 - E5b';
                    fprintf('PRN E%i will calculate TEC with %s data \n',i,PRNdata(i).Measure)
        if any(strcmp(VarNames,"S1X")) && any(strcmp(VarNames,"S7X"))
                        PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["S1X","S7X"],["SF1","SF2"]);
        end
        elseif any(strcmp(VarNames,"C1C")) && any(strcmp(VarNames,"C6C")) && any(strcmp(VarNames,"L1C")) && any(strcmp(VarNames,"L6C"))
                    PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["C1C","C6C","L1C","L6C"],["CF1","CF2","LF1","LF2"]);
                    PRNdata(i).F1 = GALE1;
                    PRNdata(i).F2 = GALE6;
                    PRNdata(i).Measure = 'E1 - E6';
                    fprintf('PRN E%i will calculate TEC with %s data \n',i,PRNdata(i).Measure)
        if any(strcmp(VarNames,"S1C")) && any(strcmp(VarNames,"S6C"))
                        PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["S1C","S6C"],["SF1","SF2"]);
        end
        elseif any(strcmp(VarNames,"C1X")) && any(strcmp(VarNames,"C6X")) && any(strcmp(VarNames,"L1X")) && any(strcmp(VarNames,"L6X"))
                    PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["C1X","C6X","L1X","L6X"],["CF1","CF2","LF1","LF2"]);
                    PRNdata(i).F1 = GALE1;
                    PRNdata(i).F2 = GALE6;
                    PRNdata(i).Measure = 'E1 - E6';
                    fprintf('PRN E%i will calculate TEC with %s data \n',i,PRNdata(i).Measure)
        if any(strcmp(VarNames,"S1X")) && any(strcmp(VarNames,"S6X"))
                        PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["S1X","S6X"],["SF1","SF2"]);
        end
        else
                    fprintf('PRN E%i does not have sufficient dual-frequency data \n',i)
                    REMOVE = [REMOVE,i];
        end
        end
        if ~isempty(REMOVE)
                sat = sat(all(sat ~= REMOVE,2));
        end
elseif system == "GLONASS"
        %% GLONASS
        fprintf("GLONASS\n")
        time = unique(data.GLONASS.Time);
        [~,f] =  min(abs(time - initialtime));
        [~,e] =  min(abs(time - endtime));
        time = time(f:e);
        [~,first] = min(abs(data.GLONASS.Time - initialtime));
        [~,last] = min(abs(data.GLONASS.Time - endtime));
        [~,I] = sort(data.GLONASS.SatelliteID(first:last));
        GLOdata = data.GLONASS(first:last,:);
        GLOdata = GLOdata(I,:);
        
        for i = 1:32
        idx = find(~(GLOdata.SatelliteID(:)-i));
        PRNTT = GLOdata(idx,:);
        PRNdata(i) = Var_Names(PRNTT);
        n = floor(p*height(PRNTT));
        PRNdata(i).PRNTT = rmmissing(PRNdata(i).PRNTT,2,'MinNumMIssing',n);
        end
        
        sat = unique(data.GLONASS.SatelliteID(first:last));
        
        %% Establish heirarchy of GLO signals to use for TEC Estimation
        % array to remove bad TEC calculation sattelites from satGPS
        REMOVE =[];
        
        for j = 1:length(sat)
            i = sat(j);
            
            VarNames = string(PRNdata(i).PRNTT.Properties.VariableNames);
        
        
        % if any(strcmp(VarNames,"C1C")) && any(strcmp(VarNames,"C3X")) && any(strcmp(VarNames,"L1C")) && any(strcmp(VarNames,"L3X"))
        %     PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["C1C","C3X","L1C","L3X"],["CF1","CF2","LF1","LF2"]);
        %     PRNdata(i).F1 = GLOL1;
        %     PRNdata(i).F2 = GLOL3;
        %     PRNdata(i).Measure = 'G1 - G3';
        %     fprintf('PRN R%i will calculate TEC with %s data \n',i,PRNdata(i).Measure)
        %     if any(strcmp(VarNames,"S1C")) && any(strcmp(VarNames,"S3X"))
        %         PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["S1C","S3X"],["SF1","SF2"]);
        %     end
        if any(strcmp(VarNames,"C1C")) && any(strcmp(VarNames,"C2C")) && any(strcmp(VarNames,"L1C")) && any(strcmp(VarNames,"L2C"))
            PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["C1C","C2C","L1C","L2C"],["CF1","CF2","LF1","LF2"]);
            PRNdata(i).F1 = GLOL1;
            PRNdata(i).F2 = GLOL2;
            PRNdata(i).Measure = 'G1 - G2';
            fprintf('PRN R%i will calculate TEC with %s data \n',i,PRNdata(i).Measure)
            if any(strcmp(VarNames,"S1C")) && any(strcmp(VarNames,"S2C"))
                PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["S1C","S2C"],["SF1","SF2"]);
            end
        else
            fprintf('PRN R%i does not have sufficient dual-frequency data \n',i)
            REMOVE = [REMOVE,i];
        end
        end

        if ~isempty(REMOVE)
        sat = sat(all(sat ~= REMOVE,2));
        end
elseif system == "BeiDou"
        %% BeiDou
        fprintf("BeiDou\n")
        time = unique(data.BeiDou.Time);
        [~,f] =  min(abs(time - initialtime));
        [~,e] =  min(abs(time - endtime));
        time = time(f:e);        [~,first] = min(abs(data.BeiDou.Time - initialtime));
        [~,last] = min(abs(data.BeiDou.Time - endtime));
        [~,I] = sort(data.BeiDou.SatelliteID(first:last));
        BDSdata = data.BeiDou(first:last,:);
        BDSdata = BDSdata(I,:);
        
        for i = 1:62
        idx = find(~(BDSdata.SatelliteID(:)-i));
        PRNTT = BDSdata(idx,:);
        PRNdata(i) = Var_Names(PRNTT);
        n = floor(p*height(PRNTT));
        PRNdata(i).PRNTT = rmmissing(PRNdata(i).PRNTT,2,'MinNumMIssing',n);
        end
        
        sat = unique(data.BeiDou.SatelliteID(first:last));
        
        %% Establish heirarchy of BDS signals to use for TEC Estimation
        % array to remove bad TEC calculation sattelites from satGPS
        REMOVE =[];
        
        for j = 1:length(sat)
            i = sat(j);
            
            VarNames = string(PRNdata(i).PRNTT.Properties.VariableNames);
        
        
        if any(strcmp(VarNames,"C2I")) && any(strcmp(VarNames,"C7Q")) && any(strcmp(VarNames,"L2I")) && any(strcmp(VarNames,"L7Q"))
            PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["C2I","C7Q","L2I","L7Q"],["CF1","CF2","LF1","LF2"]);
            PRNdata(i).F1 = BDSB1;
            PRNdata(i).F2 = BDSB2;
            PRNdata(i).Measure = 'B1 - B2Q';
            fprintf('PRN C%i will calculate TEC with %s data \n',i,PRNdata(i).Measure) 
            if any(strcmp(VarNames,"S2I")) && any(strcmp(VarNames,"S7Q"))
                PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["S2I","S7Q"],["SF1","SF2"]);
            end
        elseif any(strcmp(VarNames,"C2I")) && any(strcmp(VarNames,"C7I")) && any(strcmp(VarNames,"L2I")) && any(strcmp(VarNames,"L7I"))
            PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["C2I","C7I","L2I","L7I"],["CF1","CF2","LF1","LF2"]);
            PRNdata(i).F1 = BDSB1;
            PRNdata(i).F2 = BDSB2;
            PRNdata(i).Measure = 'B1 - B2I';
            fprintf('PRN C%i will calculate TEC with %s data \n',i,PRNdata(i).Measure)
            if any(strcmp(VarNames,"S2I")) && any(strcmp(VarNames,"S7I"))
                PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["S2I","S7I"],["SF1","SF2"]);
            end
        elseif any(strcmp(VarNames,"C2I")) && any(strcmp(VarNames,"C7X")) && any(strcmp(VarNames,"L2I")) && any(strcmp(VarNames,"L7X"))
            PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["C2I","C7X","L2I","L7X"],["CF1","CF2","LF1","LF2"]);
            PRNdata(i).F1 = BDSB1;
            PRNdata(i).F2 = BDSB2;
            PRNdata(i).Measure = 'B1 - B2X';
            fprintf('PRN C%i will calculate TEC with %s data \n',i,PRNdata(i).Measure)
            if any(strcmp(VarNames,"S2I")) && any(strcmp(VarNames,"S7X"))
                PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["S2I","S7X"],["SF1","SF2"]);
            end
        elseif any(strcmp(VarNames,"C2I")) && any(strcmp(VarNames,"C6I")) && any(strcmp(VarNames,"L2I")) && any(strcmp(VarNames,"L6I"))
            PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["C2I","C6I","L2I","L6I"],["CF1","CF2","LF1","LF2"]);
            PRNdata(i).F1 = BDSB1;
            PRNdata(i).F2 = BDSB3;
            PRNdata(i).Measure = 'B1 - B3I';
            fprintf('PRN C%i will calculate TEC with %s data \n',i,PRNdata(i).Measure)
              if any(strcmp(VarNames,"S2I")) && any(strcmp(VarNames,"S6I"))
                PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["S2I","S6I"],["SF1","SF2"]);
              end
         % elseif any(strcmp(VarNames,"C1P")) && any(strcmp(VarNames,"C2I")) && any(strcmp(VarNames,"L2I")) && any(strcmp(VarNames,"L1P"))
         %    PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["C1P","C2I","L1P","L2I"],["CF1","CF2","LF1","LF2"]);
         %    PRNdata(i).F1 = GPSL1;
         %    PRNdata(i).F2 = BDSB1;
         %    PRNdata(i).Measure = 'L1 - B1';
         %    fprintf('PRN C%i will calculate TEC with %s data \n',i,PRNdata(i).Measure)
         %    if any(strcmp(VarNames,"S1P")) && any(strcmp(VarNames,"S2I"))
         %        PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["S1P","S2I"],["SF1","SF2"]);
         %    end
        elseif any(strcmp(VarNames,"C2I")) && any(strcmp(VarNames,"C5P")) && any(strcmp(VarNames,"L2I")) && any(strcmp(VarNames,"L5P"))
            PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["C2I","C5P","L2I","L5P"],["CF1","CF2","LF1","LF2"]);
            PRNdata(i).F1 = BDSB1;
            PRNdata(i).F2 = BDSB2a;
            PRNdata(i).Measure = 'B1 - B2a';
            fprintf('PRN C%i will calculate TEC with %s data \n',i,PRNdata(i).Measure)
            if any(strcmp(VarNames,"S2I")) && any(strcmp(VarNames,"S5P"))
                PRNdata(i).PRNTT = renamevars(PRNdata(i).PRNTT,["S2I","S5P"],["SF1","SF2"]);
            end
        else
            fprintf('PRN C%i does not have sufficient dual-frequency data \n',i)
            REMOVE = [REMOVE,i];
        end
        end
        
        if ~isempty(REMOVE)
        sat = sat(all(sat ~= REMOVE,2));
        end

else
    fprintf('Incompatible system selection\n')
end

end

function [nPRNdata,gPRNdata,lPRNdata,bPRNdata] = satelliteBiasSinex(filename,nPRNdata,gPRNdata,lPRNdata,bPRNdata)
fid = fopen(filename);

    for i = 1:length(nPRNdata)
        nPRNdata(i).SatBias.obs = [];
        nPRNdata(i).SatBias.offset = [];
    end
    for i = 1:length(gPRNdata)
        gPRNdata(i).SatBias.obs = [];
        gPRNdata(i).SatBias.offset = [];
    end
    for i = 1:length(lPRNdata)
        lPRNdata(i).SatBias.obs = [];
        lPRNdata(i).SatBias.offset = [];
    end
    for i = 1:length(bPRNdata)
        bPRNdata(i).SatBias.obs = [];
        bPRNdata(i).SatBias.offset = [];
    end

current_line = fgetl(fid);
    while ~startsWith(current_line,' DSB')
        current_line = fgetl(fid);
    end

    while startsWith(current_line,' DSB')
        system = current_line(12);
        sat = str2double(current_line(13:14));
        OBS1 = current_line(26:28);
        OBS2 = current_line(31:33);
        Value = current_line(85:91);
        %st_dev = current_line(98:103);
        switch system
            case 'G'
                nPRNdata(sat).SatBias.obs = [nPRNdata(sat).SatBias.obs; string(strcat(OBS1,'-',OBS2))];
                nPRNdata(sat).SatBias.offset = [nPRNdata(sat).SatBias.offset; str2double(Value)];
            case 'R'
                gPRNdata(sat).SatBias.obs = [gPRNdata(sat).SatBias.obs; string(strcat(OBS1,'-',OBS2))];
                gPRNdata(sat).SatBias.offset = [gPRNdata(sat).SatBias.offset; str2double(Value)];
            case 'E'
                lPRNdata(sat).SatBias.obs = [lPRNdata(sat).SatBias.obs; string(strcat(OBS1,'-',OBS2))];
                lPRNdata(sat).SatBias.offset = [lPRNdata(sat).SatBias.offset; str2double(Value)];
            case 'C'
                bPRNdata(sat).SatBias.obs = [bPRNdata(sat).SatBias.obs; string(strcat(OBS1,'-',OBS2))];
                bPRNdata(sat).SatBias.offset = [bPRNdata(sat).SatBias.offset; str2double(Value)];
            otherwise
                break
        end
       

     if contains(current_line,'ENDBIA')
        break;
     end
        current_line = fgetl(fid);
    end

end

