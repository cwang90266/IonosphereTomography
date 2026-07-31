clc;close all;clear
%% Ionosonde Plot Dynasonde electron-density proionosonde_datas every 30 minutes
ionosonde_data = '/Users/User2/Desktop/PlanetiQ/slides/Jul29_26/tromso_dynasonde_2025-12-15.nc';
timeUnix = ncread(ionosonde_data, 'timestamps');
altitude = ncread(ionosonde_data, 'gdalt');       % km
pf       = ncread(ionosonde_data, 'pf');          % MHz

timeUTC = datetime(timeUnix, ...
    'ConvertFrom', 'posixtime', ...
    'TimeZone', 'UTC');
% Convert plasma frequency to Ne
% fp [Hz] = 8.98 * sqrt(Ne [m^-3])

Ne = (pf * 1e6 / 8.98).^2;       % electrons/m^3

Ne(~isfinite(Ne)) = NaN;
altitude(~isfinite(altitude)) = NaN;

% select the representative prof that closest to every half hr
startTime = dateshift(timeUTC(1), 'start', 'hour');
endTime   = timeUTC(end);

thirtymin_window = startTime : minutes(30) : endTime;

selectedIndex = zeros(size(thirtymin_window));

for k = 1:numel(thirtymin_window)
    [~, selectedIndex(k)] = min(abs(timeUTC - thirtymin_window(k)));
end

% Avoid duplicated proionosonde_datas
selectedIndex = unique(selectedIndex, 'stable');

%% Plot
figure('Position', [100 100 900 700]);


subplot(1,2,1)
hold on;
cmap = turbo(numel(selectedIndex));

for k = 1:numel(selectedIndex)

    idx = selectedIndex(k);

    selected_ionosonde_data = Ne(:, idx);

    valid = isfinite(altitude) & ...
        isfinite(selected_ionosonde_data);

    if nnz(valid) < 2
        continue
    end

    alt_temp = altitude(valid);
    ne_temp  = selected_ionosonde_data(valid);

    [alt_temp, sortIndex] = sort(alt_temp);
    ne_temp = ne_temp(sortIndex);

    plot(ne_temp, alt_temp, ...
        'LineWidth', 2, ...
        'Color', cmap(k, :), ...
        'DisplayName', datestr(timeUTC(idx), 'HH:MM'));
end

hold off;
grid on;

xlabel('N_e (m^{-3})');
ylabel('Altitude (km)');

title({
    'Tromsø Dynasonde Ne'
    '15 December 2025'
    });

ylim([50 850]);
xlim([0 inf]);

legend('NumColumns', 3, ...
    'FontSize', 15);

set(gca, 'FontSize', 20);
%% ISR
isr_data = '/Users/User2/Desktop/PlanetiQ/slides/Jul29_26/MAD6400_2025-12-15_beata_60@uhfa.nc';
ncdisp(isr_data)

isr_time = ncread(isr_data,'timestamps');
isr_range = ncread(isr_data,'range');
isr_alt = ncread(isr_data,'gdalt');
isr_ne = ncread(isr_data,'ne');
isr_dne = ncread(isr_data,'dne');

% Convert Unix time to UTC datetime
isr_datetime = datetime(isr_time, ...
    'ConvertFrom', 'posixtime', ...
    'TimeZone', 'UTC');

% Basic quality control
isr_ne(~isfinite(isr_ne)) = NaN;
isr_dne(~isfinite(isr_dne)) = NaN;
isr_alt(~isfinite(isr_alt)) = NaN;

% Select nearest profile every 30 minutes
requested_time = dateshift(isr_datetime(1), 'start', 'hour') : ...
    minutes(30) : ...
    isr_datetime(end);

isr_index = zeros(size(requested_time));

for ii = 1:numel(requested_time)
    [~, isr_index(ii)] = min(abs(isr_datetime - requested_time(ii)));
end

% Remove duplicated indices
isr_index = unique(isr_index, 'stable');

% Plot ISR profiles
subplot(1,2,2)
hold on

cmap = turbo(numel(isr_index));

for ii = 1:numel(isr_index)

    idx = isr_index(ii);

    alt_profile = isr_alt(:,idx);
    ne_profile  = isr_ne(:,idx);
    dne_profile = isr_dne(:,idx);

    % Keep valid ISR measurements
    valid = isfinite(alt_profile) & ...
        isfinite(ne_profile) & ...
        ne_profile > 0;

    if nnz(valid) < 2
        continue
    end

    alt_plot = alt_profile(valid);
    ne_plot  = ne_profile(valid);
    dne_plot  = dne_profile(valid);

    % Sort by altitude
    [alt_plot, sort_idx] = sort(alt_plot);
    ne_plot = ne_plot(sort_idx);
    dne_plot = dne_plot(sort_idx);

    plot(ne_plot-dne_plot, alt_plot, ...
        'LineWidth', 2, ...
        'Color', cmap(ii,:), ...
        'DisplayName', datestr(isr_datetime(idx), 'HH:MM'));
end

hold off
grid on
box on

xlabel('N_e (m^{-3})')
ylabel('Altitude (km)')
title('EISCAT Tromsø ISR')

ylim([50 850])
xlim([0 inf])

set(gca,'FontSize', 20)

legend('NumColumns', 3,'FontSize', 15)
%% ISR sampling rate
figure;
hist(diff(isr_time))
xlabel('Interval Between Consecutive Measurements')
ylabel('Count')
title({'Example of ISR Sampling Rate',[datestr(isr_datetime(1), 'yyyy-mm-dd HH:MM') '~' datestr(isr_datetime(end), 'yyyy-mm-dd HH:MM')]})
set(gca,'FontSize', 20)
%%
clear;
% ============================================================
%  Dynasonde data
% =============================================================

ionosonde_data = ...
    '/Users/User2/Desktop/PlanetiQ/slides/Jul29_26/tromso_dynasonde_2025-12-16.nc';

dyn_time_unix = ncread(ionosonde_data, 'timestamps');
dyn_alt       = ncread(ionosonde_data, 'gdalt');    % km
dyn_pf        = ncread(ionosonde_data, 'pf');       % MHz

% Convert Unix time to UTC
dyn_time = datetime(dyn_time_unix, ...
    'ConvertFrom', 'posixtime', ...
    'TimeZone', 'UTC');

% Convert plasma frequency to electron density
% fp [Hz] = 8.98*sqrt(Ne [m^-3])
dyn_ne = (dyn_pf*1e6/8.98).^2;                      % m^-3

% Remove invalid values
dyn_alt(~isfinite(dyn_alt)) = NaN;
dyn_ne(~isfinite(dyn_ne) | dyn_ne <= 0) = NaN;


% ============================================================
%  EISCAT UHF ISR data
% =============================================================

isr_data = ...
    '/Users/User2/Desktop/PlanetiQ/slides/Jul29_26/MAD6400_2025-12-16_beata_60@uhfa.nc';

isr_time_unix = ncread(isr_data, 'timestamps');
isr_alt       = ncread(isr_data, 'gdalt');          % km
isr_ne        = ncread(isr_data, 'ne');             % m^-3
isr_dne       = ncread(isr_data, 'dne');            % m^-3

% Convert Unix time to UTC
isr_time = datetime(isr_time_unix, ...
    'ConvertFrom', 'posixtime', ...
    'TimeZone', 'UTC');

% Remove invalid values
isr_alt(~isfinite(isr_alt)) = NaN;
isr_ne(~isfinite(isr_ne) | isr_ne <= 0) = NaN;
isr_dne(~isfinite(isr_dne) | isr_dne < 0) = NaN;


% ============================================================
%  Common comparison times
% =============================================================

comparison_time = ...
    datetime(2025,12,16,4,00,0,'TimeZone','UTC') : ...
    minutes(60) : ...
    datetime(2025,12,16,9,00,0,'TimeZone','UTC');

number_times = numel(comparison_time);

% Maximum allowed difference from requested time
maximum_time_difference = minutes(20);


% ============================================================
%  Plot settings
% =============================================================

figure('Position',[50 50 1500 1100]);

t = tiledlayout(1,6, ...
    'TileSpacing','compact', ...
    'Padding','compact');

dyn_color = [0.8500 0.3250 0.0980];
isr_color = [0 0.4470 0.7410];

x_axis_limit = [1e7 1e13];
y_axis_limit = [50 700];


% ============================================================
%  Compare profiles at each time
% =============================================================

for k = 1:number_times

    target_time = comparison_time(k);

    % Find nearest Dynasonde profile
    [dyn_time_difference,dyn_index] = ...
        min(abs(dyn_time-target_time));

    % Find nearest ISR profile
    [isr_time_difference,isr_index] = ...
        min(abs(isr_time-target_time));


    % Create subplot
    ax = nexttile;
    ax.XScale = 'log';
    hold(ax,'on')

    % --------------------------------------------------------
    %  ISR profile
    % ---------------------------------------------------------

    isr_handle = gobjects(1);

    if isr_time_difference <= maximum_time_difference

        % plot a few
        for plot_few = -5:5
            try
                isr_alt_profile = isr_alt(:,isr_index+plot_few);
                isr_ne_profile  = isr_ne(:,isr_index+plot_few);
                isr_dne_profile = isr_dne(:,isr_index+plot_few);

                isr_valid = ...
                    isfinite(isr_alt_profile) & ...
                    isfinite(isr_ne_profile);

                if nnz(isr_valid) >= 1

                    isr_alt_plot = isr_alt_profile(isr_valid);
                    isr_ne_plot  = isr_ne_profile(isr_valid);
                    isr_dne_plot  = isr_dne_profile(isr_valid);

                    % Sort profile by altitude
                    [isr_alt_plot,sort_index] = sort(isr_alt_plot);
                    isr_ne_plot = isr_ne_plot(sort_index);
                    isr_dne_plot = isr_dne_plot(sort_index);

                    isr_handle = plot(ax, ...
                        isr_ne_plot-isr_dne_plot, ...
                        isr_alt_plot, ...
                        '-', ...
                        'Color',isr_color, ...
                        'LineWidth',1, ...
                        'DisplayName','ISR');
                end

            catch

            end
        end
    end

    % --------------------------------------------------------
    %  Dynasonde profile
    % ---------------------------------------------------------

    dyn_handle = gobjects(1);

    if dyn_time_difference <= maximum_time_difference


        % plot few again
        for plot_few = -5:5
            try
                dyn_alt_profile = dyn_alt;
                dyn_ne_profile  = dyn_ne(:,dyn_index+plot_few);

                dyn_valid = ...
                    isfinite(dyn_alt_profile) & ...
                    isfinite(dyn_ne_profile);

                if nnz(dyn_valid) >= 1

                    dyn_alt_plot = dyn_alt_profile(dyn_valid);
                    dyn_ne_plot  = dyn_ne_profile(dyn_valid);

                    % Sort profile by altitude
                    [dyn_alt_plot,sort_index] = sort(dyn_alt_plot);
                    dyn_ne_plot = dyn_ne_plot(sort_index);

                    dyn_handle = plot(ax, ...
                        dyn_ne_plot, ...
                        dyn_alt_plot, ...
                        '-', ...
                        'Color',dyn_color, ...
                        'LineWidth',1, ...
                        'DisplayName','Ionosonde');
                end
            catch
            end
        end
    end

    % --------------------------------------------------------
    %  Subplot formatting
    % ---------------------------------------------------------

    grid(ax,'on')
    box(ax,'on')

    xlim(ax,x_axis_limit)
    ylim(ax,y_axis_limit)
    
    ax.XScale = 'log';
    xticks(ax,[1e7 1e8 1e9 1e10 1e11 1e12 1e13])
    xticklabels(ax,{...
        '10^7',...
        '10^8',...
        '10^9',...
        '10^{10}',...
        '10^{11}',...
        '10^{12}',...
        '10^{13}'})

    title_text = string(target_time,'HH:mm') + " UT";
    title(ax,title_text,'FontSize',17)

    ax.FontSize = 20;
    ax.LineWidth = 2;

    % Scientific notation on x-axis
    ax.XAxis.Exponent = 12;

    % Y labels only on left column
    if k== 1
        ylabel(ax,'Altitude (km)')
        xlabel(ax,'N_e (m^{-3})')
    end

    % Add actual observation times
    try
        dyn_actual = string(dyn_time(dyn_index-5),'HH:mm:ss');
        isr_actual = string(isr_time(isr_index-5),'HH:mm:ss');
        dyn_actualend = string(dyn_time(dyn_index+5),'HH:mm:ss');
        isr_actualend = string(isr_time(isr_index+5),'HH:mm:ss');

    catch
        try
            dyn_actual = string(dyn_time(dyn_index-5),'HH:mm:ss');
            isr_actual = string(isr_time(isr_index-5),'HH:mm:ss');
            dyn_actualend = string(dyn_time(dyn_index),'HH:mm:ss');
            isr_actualend = string(isr_time(isr_index),'HH:mm:ss');

        catch
            dyn_actual = string(dyn_time(dyn_index),'HH:mm:ss');
            isr_actual = string(isr_time(isr_index),'HH:mm:ss');
            dyn_actualend = string(dyn_time(dyn_index+5),'HH:mm:ss');
            isr_actualend = string(isr_time(isr_index+5),'HH:mm:ss');

        end
    end
    subtitle_text = ...
        {["Ion: " + dyn_actual + '~' + dyn_actualend], ...
        ["ISR: " + isr_actual + '~' + isr_actualend]};
    subtitle(ax,subtitle_text,'FontSize',15)

    % Legend only in first subplot
    if k == 1

        valid_handles = [dyn_handle isr_handle];
        valid_handles = valid_handles(isgraphics(valid_handles));

        if ~isempty(valid_handles)
            legend(ax,valid_handles, ...
                'Location','northeast', ...
                'FontSize',15)
        end
    end

    hold(ax,'off')
end


% ============================================================
%  Overall title
% =============================================================

title(t, ...
    {'Tromsø Ionosonde and UHF ISR Comparison', ...
    '16 December 2025'}, ...
    'FontSize',22, ...
    'FontWeight','bold');
%% ISR and ionosonde locations
clear
close all

load coastlines.mat     % variables: lat, long

% Instrument locations
dyn_lat = 69.662;
dyn_lon = 18.940;

isr_lat = 69.583;
isr_lon = 19.210;

% Svalbard Dynasonde (Longyearbyen)
dyn_sva_lat = 78.148;
dyn_sva_lon = 16.043;

% EISCAT Svalbard Radar (ESR)
isr_sva_lat = 78.153;
isr_sva_lon = 16.029;

figure('Position',[100 100 1200 500])

% Regional view
subplot(1,2,1)

plot(coastlon,coastlat,'k','LineWidth',1)
hold on

plot(dyn_lon,dyn_lat,'ro',...
    'MarkerFaceColor','r',...
    'MarkerSize',10)

plot(isr_lon,isr_lat,'b^',...
    'MarkerFaceColor','b',...
    'MarkerSize',10)

plot(dyn_sva_lon,dyn_sva_lat,'ro',...
    'MarkerFaceColor','r',...
    'MarkerSize',10)

plot(isr_sva_lon,isr_sva_lat,'b^',...
    'MarkerFaceColor','b',...
    'MarkerSize',10)

text(dyn_lon+0.08,dyn_lat,...
    'Dynasonde','Color','r','FontSize',12)

text(isr_lon+0.08,isr_lat,...
    'EISCAT UHF','Color','b','FontSize',12)

text(dyn_sva_lon+0.08,dyn_sva_lat,...
    'Dynasonde','Color','r','FontSize',12)

text(isr_sva_lon+0.08,isr_sva_lat,...
    'EISCAT UHF','Color','b','FontSize',12)

axis equal
xlim([15 25])
ylim([67 71])

xlabel('Longitude (°)')
ylabel('Latitude (°)')

grid on
box on
title('Northern Norway')

% Zoom around Tromsø
subplot(1,2,2)

plot(coastlon,coastlat,'k','LineWidth',1)
hold on

plot(dyn_lon,dyn_lat,'ro',...
    'MarkerFaceColor','r',...
    'MarkerSize',10)

plot(isr_lon,isr_lat,'b^',...
    'MarkerFaceColor','b',...
    'MarkerSize',10)

plot([dyn_lon isr_lon],...
    [dyn_lat isr_lat],...
    'k--','LineWidth',1.5)

text(dyn_lon+0.01,dyn_lat,...
    'Dynasonde','Color','r','FontSize',12)

text(isr_lon+0.01,isr_lat,...
    'ISR','Color','b','FontSize',12)

axis equal
xlim([18.7 19.4])
ylim([69.45 69.75])

xlabel('Longitude (°)')
ylabel('Latitude (°)')

grid on
box on
title('Tromsø')

sgtitle('Locations of Tromsø Dynasonde and EISCAT UHF ISR')