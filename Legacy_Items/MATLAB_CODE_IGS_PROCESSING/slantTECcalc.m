function [GPS_out,GAL_out,GLO_out,BDS_out,NeQuickTEC] = slantTECcalc(initialtime,endtime,RINEX,label, inputNav, lla, stationName, isIGS, isDynamic, PVT_dynamics)
%% Rinex Processing Input Files
% Handle optional inputs with nargin defaults
if nargin < 10
    PVT_dynamics = [];
end
if nargin < 9
    isDynamic = false;
end
if nargin < 8
    isIGS = false;
end

NeQuickTEC = [];
doy = day(datetime(initialtime),'dayofyear');

% % Input Rinex Observation File
% inputObsname = strcat("data/janf",num2str(doy),"0.24o");
% Sinex satellite IFB file
inputSinex = strcat("DATA/CAS0OPSRAP_2025",num2str(doy),"0000_01D_01D_DCB.BIA"); % https://cddis.nasa.gov/archive/gnss/products/bias/2024/
% Galileo Nav File
GalNav = strcat("DATA/AMC400USA_R_2025",num2str(doy),"0000_01D_EN.rnx"); %https://cddis.nasa.gov/archive/gnss/data/daily/2024/
% Mixed Nav file (from Rx)
% inputNav = strcat("DATA/JANF",num2str(doy),"0.25N");


c = 299792458; %m/s

% Define the timeframe

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

sigmaBounds = 5;
r_e = 6378000;
r_h = 250000;
%% 


% Frequency Selection and Carrier Adjusted Pseudorange: GPS
[nPRNdata, satGPS,timeGPS] = frequencySelectionRinex(RINEX,"GPS", initialtime, endtime);
nPRNdata = cycleSlipRepair(nPRNdata,satGPS,sigmaBounds);
% nPRNdata(16).Measure = [];
[nPRNdata, satGPS] = carrierPhaseLeveling(nPRNdata,satGPS);
% 
% % Frequency Selection and Carrier Adjusted Pseudorange: Galileo
[lPRNdata, satGAL,timeGAL] = frequencySelectionRinex(RINEX,"Galileo", initialtime, endtime);
lPRNdata = cycleSlipRepair(lPRNdata,satGAL,sigmaBounds);
[lPRNdata, satGAL] = carrierPhaseLeveling(lPRNdata,satGAL);
% 
% % Frequency Selection and Carrier Adjusted Pseudorange: GLONASS
[gPRNdata, satGLO,timeGLO] = frequencySelectionRinex(RINEX,"GLONASS", initialtime, endtime);
gPRNdata = cycleSlipRepair(gPRNdata,satGLO,7*sigmaBounds);
[gPRNdata, satGLO] = carrierPhaseLeveling(gPRNdata,satGLO);
% 
% % Frequency Selection and Carrier Adjusted Pseudorange: BeiDou
[bPRNdata, satBDS,timeBDS] = frequencySelectionRinex(RINEX,"BeiDou", initialtime, endtime);
bPRNdata = cycleSlipRepair(bPRNdata,satBDS,sigmaBounds);
[bPRNdata, satBDS] = carrierPhaseLeveling(bPRNdata,satBDS);

%% Sinex Dual Frequency Satellite IFB

    % legendtxt = [legendtxt, '',strcat("GPS: ",num2str(satGPS(j)))];
    figure(200)%satGPS(i))
    subplot(2,2,1); hold on;
    for j = 1:length(satGPS)
        i = satGPS(j);
        if i ~= 24
            continue
        end
        plot(nPRNdata(i).PRNTT.Time,nPRNdata(i).PRNTT.PseudoDiff,'k:',LineWidth=0.5)
        R = plot(nPRNdata(i).PRNTT.Time,nPRNdata(i).PRNTT.PseudoDiff_cor,'b.',linewidth=1);
    end
    subplot(2,2,2); hold on;
    for j = 1:length(satGAL)
        i = satGAL(j);
        plot(lPRNdata(i).PRNTT.Time,lPRNdata(i).PRNTT.PseudoDiff,'k:',LineWidth=0.5)
        R = plot(lPRNdata(i).PRNTT.Time,lPRNdata(i).PRNTT.PseudoDiff_cor,'b.',linewidth=1);
    end
    subplot(2,2,3); hold on;
    for j = 1:length(satGLO)
        i = satGLO(j);
        plot(gPRNdata(i).PRNTT.Time,gPRNdata(i).PRNTT.PseudoDiff,'k:',LineWidth=0.5)
        R = plot(gPRNdata(i).PRNTT.Time,gPRNdata(i).PRNTT.PseudoDiff_cor,'b.',linewidth=1);
    end
    subplot(2,2,4); hold on;
    for j = 1:length(satBDS)
        i = satBDS(j);
        plot(bPRNdata(i).PRNTT.Time,bPRNdata(i).PRNTT.PseudoDiff,'k:',LineWidth=0.5)
        R = plot(bPRNdata(i).PRNTT.Time,bPRNdata(i).PRNTT.PseudoDiff_cor,'b.',linewidth=1);
    end


% Replace the four separate calls in slantTECcalc.m with one:
[nPRNdata, gPRNdata, lPRNdata, bPRNdata] = satelliteBiasSinex(inputSinex, ...
    nPRNdata, gPRNdata, lPRNdata, bPRNdata);

    % legendtxt = [legendtxt, '',strcat("GPS: ",num2str(satGPS(j)))];
    figure(gcf)%satGPS(i))
    subplot(2,2,1); hold on;
    for j = 1:length(satGPS)
        i = satGPS(j);
        if i ~= 24
            continue
        end
        COR = plot(nPRNdata(i).PRNTT.Time,nPRNdata(i).PRNTT.PseudoDiff_cor,'g.',linewidth=1);  
    end
    subplot(2,2,2); hold on;
    for j = 1:length(satGAL)
        i = satGAL(j);
        COR = plot(lPRNdata(i).PRNTT.Time,lPRNdata(i).PRNTT.PseudoDiff_cor,'g.',linewidth=1); 
    end
    subplot(2,2,3); hold on;
    for j = 1:length(satGLO)
        i = satGLO(j);
        COR = plot(gPRNdata(i).PRNTT.Time,gPRNdata(i).PRNTT.PseudoDiff_cor,'g.',linewidth=1);
    end
    subplot(2,2,4); hold on;
    for j = 1:length(satBDS)
        i = satBDS(j);
        COR = plot(bPRNdata(i).PRNTT.Time,bPRNdata(i).PRNTT.PseudoDiff_cor,'g.',linewidth=1);
    end


        figure(gcf)
        const = ["GPS", "GAL", "GLO", "BDS"];
        for sub = 1:4
           subplot(2,2,sub);
            legend([R,COR],{"No Satellite Bias Adjust","Sinex Bias Adjusted"});
            ylabel("m")
            title(const(sub))
            grid on;
 
        end
        sgtitle("Sinex Adjustment")

        
% Find the PRN with the largest sequential jump in PseudoDiff_cor for each constellation

constellations = {
    struct('name',"GPS", 'block',nPRNdata, 'sats',satGPS, 'color',[0 0.4470 0.7410])
    struct('name',"GAL", 'block',lPRNdata, 'sats',satGAL, 'color',[0.8500 0.3250 0.0980])
    struct('name',"GLO", 'block',gPRNdata, 'sats',satGLO, 'color',[0.9290 0.6940 0.1250])
    struct('name',"BDS", 'block',bPRNdata, 'sats',satBDS, 'color',[0.4940 0.1840 0.5560])
};

% Pre-allocate one best-hit struct per constellation
best(4) = struct('idx',[], 'prn',[], 'maxJump',-inf, 'Time',[], 'Y',[], 'jumpLoc',[]);


for c = 1:4
    best(c).maxJump = -inf;
    blk   = constellations{c}.block;
    names = constellations{c}.sats;

    for k = 1:length(names)
        i = names(k);
        T = blk(i).PRNTT.Time;
        Y = blk(i).PRNTT.PseudoDiff_cor;

        if isempty(T) || isempty(Y), continue; end
        valid = isfinite(Y) & ~isnat(T);
        if nnz(valid) < 2,           continue; end

        [Tsort, ord] = sort(T(valid));
        Ysort        = Y(valid);
        Ysort        = Ysort(ord);

        jumps   = abs(diff(Ysort));
        maxJump = max(jumps);

        if maxJump > best(c).maxJump
            best(c).idx     = i;
            best(c).prn     = names(k);
            best(c).maxJump = maxJump;
            best(c).Time    = Tsort;
            best(c).Y       = Ysort;
            [~, best(c).jumpLoc] = max(jumps);   % store index of largest jump
        end
    end
end

% ── Plot ────────────────────────────────────────────────────────────────────
figure(300); clf;
tiledlayout(2, 2, 'TileSpacing','compact', 'Padding','compact');

for c = 1:4
    nexttile;

    cname = constellations{c}.name;
    col   = constellations{c}.color;
    b     = best(c);

    if isempty(b.idx)
        text(0.5, 0.5, sprintf('%s\nNo valid data', cname), ...
            'HorizontalAlignment','center', 'Units','normalized', 'FontSize',11);
        title(cname);
        continue
    end

    % Main series
    plot(b.Time, b.Y, '.-', 'Color',col, 'LineWidth',1, 'MarkerSize',8, ...
        'DisplayName','PseudoDiff\_cor');
    hold on; grid on;

    % Highlight the largest jump
    loc  = b.jumpLoc;
    idx1 = loc;
    idx2 = loc + 1;
    plot(b.Time([idx1,idx2]), b.Y([idx1,idx2]), 'o-', ...
        'Color','r', 'LineWidth',2, 'MarkerSize',9, ...
        'DisplayName',sprintf('Largest jump (%.3f m)', b.maxJump));

    xlabel('Time');
    ylabel('PseudoDiff\_cor (m)');
    title(sprintf('%s  –  PRN %d  |  max jump = %.3f m', cname, b.prn, b.maxJump));
    legend('Location','best');
end

sgtitle('Largest Sequential PseudoDiff\_cor Jump per Constellation', 'FontWeight','bold');
%% NeQuick Model: IPP and VTEC

% lla = [40.01001839149076, -105.24402290241993, 1648.53];

% Determine the NeQuick Coefficients
info = rinexinfo(GalNav);
coef = info.IonosphericCorrections.Parameters(1:3);

UTC = datevec(timeGPS);
RxPos = ones(length(UTC),3).*lla;
SatPos = ones(length(UTC),3).*lla;
SatPos(:,3) = 202e5;

% Determine Vertical TEC 

    alt = linspace(RxPos(1,3),20e5,length(timeGPS));
    RxPos(:,3) = alt;
    utc = ones(size(UTC)).*UTC(1,:);
    [~,tec] = calcModel(utc,RxPos,SatPos,1,coef,"height");
    
    density = -gradient(tec)./gradient(alt)';
    density = smooth(density,10,'rloess');
    [~, maxAlt] = max(density);
    %% Plots of IPP Calculation
    figure
    subplot(2,1,1)
    hold on;
    plot(alt./1e3,tec)
    title("TEC vs Receiver Altitude")
    ylabel("TEC")
    xlabel("Altitude (km)")
    subplot(2,1,2)
    hold on;
    plot(alt./1e3,density);
    linelabel = strcat("Max Density: ",num2str(alt(maxAlt)/1e3)," km");
    xline(alt(maxAlt)./1e3,'-',linelabel,"LabelVerticalAlignment","bottom");
    ylabel("TEC/m")
    xlabel("Altitude (km)")
    %%
IPP = alt(maxAlt);
% IPP = 450;

%% VTEC and Bias Calculation
fprintf("VTEC_B\n")
% load("nprn_postgeo.mat","nPRNdata");
% Repeat VTEC_B_geo processing for all 4 constellations and cache GPS result
fn = sprintf("nprn_postgeo_new_%s.mat",label);
if isfile(fn)
    S = load(fn,"nPRNdata","satPosGPS");
    if isfield(S,"nPRNdata"), nPRNdata = S.nPRNdata; end
    if isfield(S,"satPosGPS"), satPosGPS = S.satPosGPS; end
else
    [nPRNdata,satPosGPS] = VTEC_B_geo(nPRNdata, satGPS, inputNav, lla, 'GPS',IPP,timeGPS);
    save(fn,"satPosGPS","nPRNdata")
end

% Galileo
fn = sprintf("lprn_postgeo_new_%s.mat",label);
if isfile(fn)
    S = load(fn,"lPRNdata","satPosGAL");
    if isfield(S,"lPRNdata"), lPRNdata = S.lPRNdata; end
    if isfield(S,"satPosGAL"), satPosGAL = S.satPosGAL; end
else
    [lPRNdata,satPosGAL] = VTEC_B_geo(lPRNdata, satGAL, inputNav, lla, 'Galileo',IPP,timeGAL);
    save(fn,"satPosGAL","lPRNdata")
end

% GLONASS
fn = sprintf("gprn_postgeo_new_%s.mat",label);
if isfile(fn)
    S = load(fn,"gPRNdata","satPosGLO");
    if isfield(S,"gPRNdata"), gPRNdata = S.gPRNdata; end
    if isfield(S,"satPosGLO"), satPosGLO = S.satPosGLO; end
else
    [gPRNdata,satPosGLO] = VTEC_B_geo(gPRNdata, satGLO, inputNav, lla, 'GLONASS',IPP,timeGLO);
    save(fn,"satPosGLO","gPRNdata")
end

% BeiDou
fn = sprintf("bprn_postgeo_new_%s.mat",label);
if isfile(fn)
    S = load(fn,"bPRNdata","satPosBDS");
    if isfield(S,"bPRNdata"), bPRNdata = S.bPRNdata; end
    if isfield(S,"satPosBDS"), satPosBDS = S.satPosBDS; end
else
    [bPRNdata,satPosBDS] = VTEC_B_geo(bPRNdata, satBDS, inputNav, lla, 'BeiDou',IPP,timeBDS);
    save(fn,"satPosBDS","bPRNdata")
end
%% Padding, VTEC Calculation, and Bias Adjustment
epochs = 4*10*2;
el = 15;

% GPS
fprintf("Processing GPS...\n")
nPRNdata_pad = nPRNdata;
% for j = 1:length(satGPS)
%     i = satGPS(j);
%     nPRNdata_pad(i).PRNTT = nPRNdata_pad(i).PRNTT(nPRNdata_pad(i).PRNTT.Time < (datetime(initialtime)+minutes(5)),:);
% end
if strcmp(label,"Flight_Data_18092025")
time_pad = timeGPS(timeGPS < '18-Sep-2025 13:00:00');
else
time_pad = timeGPS;
end
sigma_c_m = 1.0;   % metres — tune between 1.0 (tight) and 10.0 (loose)

%%
[nPRNdata, lPRNdata, gPRNdata, bPRNdata, TECANS, Bias, K_all] = ...
    VTEC_MultiConstellation_calc(nPRNdata, lPRNdata, gPRNdata, bPRNdata, ...
                                  satGPS, satGAL, satGLO, satBDS, ...
                                  time_pad, epochs, el, coef, lla, ...
                                  sigma_c_m, label, ...
                                  inputSinex, stationName, isIGS);

%%

% GPS
% el_g = initial_el;
% while true
%     [nPRNdata,TECANS_GPS,Bias_GPS,K_GPS] = VTEC_B_calc(nPRNdata,satGPS,time_pad,epochs,el_g,"GPS",coef,lla,sigma_c_m,label);
%     if any(Bias_GPS(:)) || el_g <= min_el
%         break
%     end
%     el_g = el_g - decr;
% end
% 
% % Galileo
% el_l = initial_el;
% while true
%     [lPRNdata,TECANS_GAL,Bias_GAL,K_GAL] = VTEC_B_calc(lPRNdata,satGAL,time_pad,epochs,el_l,"Galileo",coef,lla,sigma_c_m,label);
%     if any(Bias_GAL(:)) || el_l <= min_el
%         break
%     end
%     el_l = el_l - decr;
% end
% 
% % GLONASS
% el_glo = initial_el;
% while true
%     [gPRNdata,TECANS_GLO,Bias_GLO,K_GLO] = VTEC_B_calc(gPRNdata,satGLO,time_pad,epochs,el_glo,"GLONASS",coef,lla,sigma_c_m,label);
%     if any(Bias_GLO(:)) || el_glo <= min_el
%         break
%     end
%     el_glo = el_glo - decr;
% end
% 
% % BeiDou
% el_b = initial_el;
% while true
%     [bPRNdata,TECANS_BDS,Bias_BDS,K_BDS] = VTEC_B_calc(bPRNdata,satBDS,time_pad,epochs,el_b,"BeiDou",coef,lla,sigma_c_m,label);
%     if any(Bias_BDS(:)) || el_b <= min_el
%         break
%     end
%     el_b = el_b - decr;
% end
% for j = 1:length(satGPS)
%     i = satGPS(j);
%     nPRNdata(i).Bias = nPRNdata_pad(i).Bias;
% end
nPRNdata = BiasAdjust(nPRNdata,satGPS);
lPRNdata = BiasAdjust(lPRNdata,satGAL);
gPRNdata = BiasAdjust(gPRNdata,satGLO);
bPRNdata = BiasAdjust(bPRNdata,satBDS);


%% NeQuick slantTECs for Kinematic/Static Receivers
fprintf("Computing epoch-by-epoch receiver PVT for NeQuick sTEC...\n")

% --- 1. Parse the receiver trajectory ---
% Replace 'your_receiver_data.csv' with your actual file name
if isDynamic
    [rxPVT_Time, rxPVT_LLA] = parseSeptentrioPVT(PVT_dynamics);
else
    
% For static receiver (or when not dynamic), build rxPVT_Time and rxPVT_LLA
% from available time vectors (timeGPS, timeGAL, timeGLO, timeBDS) and use
% receiver LLA (lla) to create matching-position entries for each time.
% Collect non-empty time vectors
timeVectors = {};
if exist('timeGPS','var') && ~isempty(timeGPS); timeVectors{end+1} = timeGPS; end
if exist('timeGAL','var') && ~isempty(timeGAL); timeVectors{end+1} = timeGAL; end
if exist('timeGLO','var') && ~isempty(timeGLO); timeVectors{end+1} = timeGLO; end
if exist('timeBDS','var') && ~isempty(timeBDS); timeVectors{end+1} = timeBDS; end

if isempty(timeVectors)
    rxPVT_Time = [];
    rxPVT_LLA = [];
else
    % Concatenate all times and get unique sorted times
    allTimes = vertcat(timeVectors{:});
    allTimes = unique(allTimes);
    rxPVT_Time = allTimes;
    % Create rxPVT_LLA with same number of rows as rxPVT_Time, repeating lla
    % Expecting lla as [lat lon height] or Nx3; handle both
    if isempty(lla)
        rxPVT_LLA = [];
    else
        if isvector(lla) && numel(lla) == 3
            rxPVT_LLA = repmat(reshape(lla,1,3), numel(rxPVT_Time), 1);
        elseif size(lla,2) == 3
            % If lla already has multiple rows, but single timestamp expected,
            % take the first row and replicate
            rxPVT_LLA = repmat(lla(1,1:3), numel(rxPVT_Time), 1);
        else
            % Fallback: try to reshape to 1x3 then replicate
            tmp = lla(:)';
            tmp = tmp(1:min(3,numel(tmp)));
            tmp = [tmp, zeros(1,3-numel(tmp))];
            rxPVT_LLA = repmat(tmp, numel(rxPVT_Time), 1);
        end
    end
end
end

% Optional: Check if data was actually loaded before running NeQuick
if isempty(rxPVT_Time)
    error('Cannot run NeQuick: No valid kinematic positions available in the trajectory file.');
end

% --- 2. Run the NeQuick prediction ---
fprintf("Calculating NeQuick sTEC for all constellations...\n")
[nPRNdata] = NeQuick_sTEC_Kinematic(nPRNdata, satPosGPS, coef, satGPS, rxPVT_Time, rxPVT_LLA);
[lPRNdata] = NeQuick_sTEC_Kinematic(lPRNdata, satPosGAL, coef, satGAL, rxPVT_Time, rxPVT_LLA);
[gPRNdata] = NeQuick_sTEC_Kinematic(gPRNdata, satPosGLO, coef, satGLO, rxPVT_Time, rxPVT_LLA);
[bPRNdata] = NeQuick_sTEC_Kinematic(bPRNdata, satPosBDS, coef, satBDS, rxPVT_Time, rxPVT_LLA);

fprintf("All systems processed.\n")



%%
% close all;
% legendtxt = [];
% for j = 1:length(satGPS)
%         i = satGPS(j);
%         % if 
%         BetaI = 1/40.3*(nPRNdata(i).F1^2 * nPRNdata(i).F2^2)/(nPRNdata(i).F1^2 - nPRNdata(i).F2^2)*10e-17;
%     legendtxt = [legendtxt, '',strcat("GPS: ",num2str(satGPS(j)))];
%     figure(10)%satGPS(i))
%     hold on;
%     plot(nPRNdata(i).PRNTT.Time,nPRNdata(i).PRNTT.PseudoDiff-nPRNdata(i).Bias,'k:',LineWidth=0.5)
%     plot(nPRNdata(i).PRNTT.Time,nPRNdata(i).PRNTT.PseudoDiff_cor-nPRNdata(i).Bias,'.',linewidth=1)
% 
% end
% 
% figure(10)
% title("Differential Ionospheric Delay - Bias Adjusted")
% xlabel("Time of Day (UTC)")
% ylabel('Delay (m)')
% grid on;
% xL = xlim; yL = ylim;
% legend(legendtxt,Location='bestoutside')
% str = {strcat("Bias: "),strcat("L1 - L2: ",num2str(Bias(1))," m"),strcat("L1 - L5: ",num2str(Bias(2))," m")};
% text(timeGPS(1),0.99*yL(2),str,'HorizontalAlignment','left','VerticalAlignment','top')

%%

figure
polaraxes;
hold on

% Define colors for each constellation
colorGPS = [0 0.4470 0.7410];    % Blue
colorGAL = [0.8500 0.3250 0.0980]; % Orange
colorGLO = [0.9290 0.6940 0.1250]; % Yellow
colorBDS = [0.4940 0.1840 0.5560]; % Purple

% Plotting GPS satellites
for j = 1:length(satGPS)
    i = satGPS(j);
    polarscatter(deg2rad(nPRNdata(i).PRNTT.az), nPRNdata(i).PRNTT.el, ...
        400, colorGPS, 'filled', 'MarkerFaceAlpha', 0.3, 'MarkerEdgeAlpha', 1);
end

% Plotting Galileo satellites
for j = 1:length(satGAL)
    k = satGAL(j);
    polarscatter(deg2rad(lPRNdata(k).PRNTT.az), lPRNdata(k).PRNTT.el, ...
        400, colorGAL, 'filled', 'MarkerFaceAlpha', 0.3, 'MarkerEdgeAlpha', 1);
end

% Plotting GLONASS satellites
for j = 1:length(satGLO)
    k = satGLO(j);
    polarscatter(deg2rad(gPRNdata(k).PRNTT.az), gPRNdata(k).PRNTT.el, ...
        400, colorGLO, 'filled', 'MarkerFaceAlpha', 0.3, 'MarkerEdgeAlpha', 1);
end

% Plotting BeiDou satellites
for j = 1:length(satBDS)
    k = satBDS(j);
    polarscatter(deg2rad(bPRNdata(k).PRNTT.az), bPRNdata(k).PRNTT.el, ...
        400, colorBDS, 'filled', 'MarkerFaceAlpha', 0.3, 'MarkerEdgeAlpha', 1);
end

ax = gca;
ax.RLim = [0 90];
ax.RDir = "reverse";
ax.ThetaDir = 'clockwise';
ax.ThetaZeroLocation = "top";

% Add legend
% legend({'GPS', 'Galileo', 'GLONASS', 'BeiDou'}, 'Location', 'best');

title('SKYPLOT')
saveas(gcf, strcat("figures\", label, "_SKYPLOT"), 'png');

close all;

% Save Output Structures

GPS_out.nPRNdata = nPRNdata;
GPS_out.satGPS = satGPS;
GPS_out.timeGPS = timeGPS;

GAL_out.lPRNdata = lPRNdata;
GAL_out.satGAL = satGAL;
GAL_out.timeGAL = timeGAL;

GLO_out.gPRNdata = gPRNdata;
GLO_out.satGLO = satGLO;
GLO_out.timeGLO = timeGLO;

BDS_out.bPRNdata = bPRNdata;
BDS_out.satBDS = satBDS;
BDS_out.timeBDS = timeBDS;


%%
function [PRNdataOUT] = PseudoDiffCalc(PRNdataIN,sat)
% Speed of Light
c = 299792458; %m/s
PRNdata = PRNdataIN;
for j = 1:length(sat)
    i = sat(j);
    BetaI = 1/40.3*(PRNdata(i).F1^2 * PRNdata(i).F2^2)/(PRNdata(i).F1^2 - PRNdata(i).F2^2)*1e-16;
    % legendtxt = [legendtxt, '',strcat("PRN: ",num2str(sat(j)))];
    PRNdata(i).PRNTT.PseudoDiff = PRNdata(i).PRNTT.CF2 - PRNdata(i).PRNTT.CF1;
    P_C = (mean(c/PRNdata(i).F1.*(PRNdata(i).PRNTT.LF1) - c/PRNdata(i).F2.*(PRNdata(i).PRNTT.LF2) - (PRNdata(i).PRNTT.CF2 - PRNdata(i).PRNTT.CF1),'all',"omitmissing"));
    if ismember('D_adr', PRNdata(i).PRNTT.Properties.VariableNames) && ~isempty(PRNdata(i).PRNTT.D_adr)
        fprintf("PRN %d is using D_adr correction \n", i)
        CarrierDiff = PRNdata(i).PRNTT.D_adr;
    else
        fprintf("PRN %d has no D_adr correction \n", i)
        CarrierDiff = c/PRNdata(i).F1.*PRNdata(i).PRNTT.LF1 - c/PRNdata(i).F2.*PRNdata(i).PRNTT.LF2;
    end
    PRNdata(i).PRNTT.PseudoDiff_cor = CarrierDiff - P_C;
    % PRNdata(i).F2.^2 /(PRNdata(i).F1^2-PRNdata(i).F2^2).*
    % diffcor = (mean((c/PRNdata(i).F1.*(PRNdata(i).PRNTT.LF1) - c/PRNdata(i).F2.*(PRNdata(i).PRNTT.LF2)),'all',"omitmissing") - mean((PRNdata(i).PRNTT.CF2 - PRNdata(i).PRNTT.CF1),'all',"omitmissing"));
    PRNdata(i).PRNTT.TECu = BetaI.*(CarrierDiff-P_C);
end
PRNdataOUT = PRNdata;
end

function [PRNdataOUT,satPos] = VTEC_B_geo(PRNdataIN, sat, filename,lla, system,IPP,time)
%% Function To Calculate VTEC and DCB given receiver data
% Implementation of Method Developed by Bourne et al.
% 
% 
% By Austin Hunter
% Read in RINEX data
disp(filename)
nav = rinexread(filename);
disp(filename)

% Speed of Light
c = 299792458; %m/s

% Earth radius and IPP
r_e = 6378000;
h_I = IPP;
PRNdata = PRNdataIN;

satPos = NaN(length(sat),length(time),3);
fprintf("Satellite Geometry Definition: \n(")
for j = 1:length(sat)
    i = sat(j);
    fprintf(" %d,",i)
    if system == "GPS"
    PRNnav = nav.GPS(nav.GPS.SatelliteID(:) == i,:);
    elseif system == "GLONASS"
    PRNnav = nav.GLONASS(nav.GLONASS.SatelliteID(:) == i,:);   
    elseif system == "Galileo"
    PRNnav = nav.Galileo(nav.Galileo.SatelliteID(:) == i,:);
    elseif system == "BeiDou"
    PRNnav = nav.BeiDou(nav.BeiDou.SatelliteID(:) == i,:);
    else
        fprintf("The %s system is not coded yet",system)
    end
for k = 1:height(PRNdata(i).PRNTT)
    [~,tidx] = min(abs(seconds(PRNnav.Time - PRNdata(i).PRNTT.Time(k))));
    [satpos,~,~] = gnssconstellation(PRNdata(i).PRNTT.Time(k),PRNnav(tidx,:));
    [PRNdata(i).PRNTT.az(k),PRNdata(i).PRNTT.el(k),PRNdata(i).PRNTT.vis(k)] = lookangles(lla,satpos,0);
    PRNdata(i).PRNTT.M_el(k) = 1/(sqrt(1 - (r_e * cosd(PRNdata(i).PRNTT.el(k))/(r_e+h_I))^2));
    PRNdata(i).PRNTT.zeta(k) = 90-PRNdata(i).PRNTT.el(k);
    PRNdata(i).PRNTT.OF(k) = (1-(r_e * sind(PRNdata(i).PRNTT.zeta(k))/(r_e +h_I))^2)^(-0.5);
    % sine rule
    PRNdata(i).PRNTT.zeta_p(k) = asind(r_e*sind(180-PRNdata(i).PRNTT.zeta(k))/(r_e+h_I));
    PRNdata(i).PRNTT.alpha(k) = 180 - PRNdata(i).PRNTT.zeta_p(k) - PRNdata(i).PRNTT.zeta(k);
    r_IPP = r_e*sind(PRNdata(i).PRNTT.alpha(k))/sind(PRNdata(i).PRNTT.zeta_p(k));
    ecef = lla2ecef(lla);
    x_IPP = ecef + r_IPP/norm(satpos).*(satpos-ecef);
    lla_IPP = ecef2lla(x_IPP);
    PRNdata(i).PRNTT.Dphi(k) = lla_IPP(1) - lla(1);
    PRNdata(i).PRNTT.Dlambda(k) = lla_IPP(2) - lla(2);
    satPos(i,k,:) = satpos;
end
    PRNdata(i).PRNTT.VTEC = PRNdata(i).PRNTT.TECu./PRNdata(i).PRNTT.M_el;
    PRNnav = [];
end
fprintf(")")

PRNdataOUT = PRNdata;
end

function PRNdata = NeQuick_sTEC(PRNdata,satPos,coef,sat,lla)
    for j = 1:length(sat)
        i = sat(j);
        k = height(PRNdata(i).PRNTT);
            UTC = datevec(PRNdata(i).PRNTT.Time);
            SatPos = ecef2lla(squeeze(satPos(i,1:k,:)));
            RxPos = ones(length(UTC),3).*lla;
            [~,PRNdata(i).PRNTT.sTEC_model] = calcModel(UTC,RxPos,SatPos,1,coef,"sTEC");
    end
end



end