function [coef] = fetchNAV(path,startDate,endDate,opts)
%FETCHNAV Fetch Galileo broadcast ephemeris data navigation files from CDDIS archives.
%
%   INPUTS:
%     path                     -   save directory
%     startDate                -   data start date
%     endDate                  -   data end date
%     opts.PrintProgress       -   print progress reports?
%
%   OUTPUTS:
%     NONE
%
%   AUTHOR:     William Gravel
%   CREATED:    03/07/2024
%   EDITED:     03/08/2024
%   PURPOSE:    GOES-R GPS code-carrier divergence research
%
%   See also FETCHSP3.

% Define arguments list
arguments (Input)
    path (1,1) string
    startDate (1,1) datetime
    endDate (1,1) datetime
    opts.PrintProgress = true
end

% Define constants
getConstants;

% Define default curl command prefix for CDDIS fetch requests
cmdPrefix = sprintf("curl -c %s/.urs_cookies -n -L -s",path,path);
cmdURL = "https://cddis.nasa.gov/archive/gnss";

% Define priority list for IGS stations
queryStations = ["USN700USA","AMC400USA","NIST00USA"];

% Get year and day of year numbers
queryYears = year(startDate:days(1):endDate);
queryDays = day(startDate:days(1):endDate,"dayofyear");

% Begin progress report
if opts.PrintProgress
    fprintf("Fetching Galileo navigation files for %s through %s...\n",startDate,endDate)
    reverseStr = '';
end

% Download RINEX files from CDDIS
for i = 1:length(queryDays)
    % Update progress report
    if opts.PrintProgress
        msg = sprintf('Downloading Galileo navigation files for Y%4d DOY %03d (%2d/%2d)... [%s%s] %3.0f%%%%',queryYears(i),queryDays(i),i,length(queryDays),repmat('#',floor(i/length(queryDays)*10),1),repmat('-',10 - floor(i/length(queryDays)*10),1),i/length(queryDays)*100);
        fprintf([reverseStr,msg]);
        reverseStr = repmat(sprintf('\b'),1,length(msg)-1);
    end

    % Get directory listing
    cmd = sprintf("%s %s/data/daily/%4d/%03d/%2dl/*_01D_EN.rnx.gz?list",cmdPrefix,cmdURL,queryYears(i),queryDays(i),mod(queryYears(i),1000));
    [~,out] = system(cmd);

    % Fetch highest priority station data file
    stationIdx = find(arrayfun(@(s) contains(out,s),queryStations),1,"first");
    if ~isnan(stationIdx)
        cmd = sprintf("%s -o %s/nav%4d%03d.rnx.gz %s/data/daily/%4d/%03d/%2dl/%s_R_%4d%03d0000_01D_EN.rnx.gz",cmdPrefix,path,queryYears(i),queryDays(i),cmdURL,queryYears(i),queryDays(i),mod(queryYears(i),1000),queryStations(stationIdx),queryYears(i),queryDays(i));
        system(cmd);
    end
end

% Extract RINEX files from downloaded archives
gunzip(fullfile(path,'*.gz'),path);
delete(fullfile(path,"*.gz"));
delete(fullfile(path,".urs_cookies"));

% End progress report
if opts.PrintProgress
    msg = ' Done.\n';
    fprintf([reverseStr,'\b',msg]);
end


end
