function [PRNTTOUT,Slip] = passCycleSlipRepair(PRNTT,F1,F2,bool)

sigmaBounds = 2;
% 1000s averaging interval
K = floor(seconds(2000)/(PRNTT.Time(2)-PRNTT.Time(1)));
samples = ceil(K/2);
fs = 1/(K);%seconds(1)/(PRNdataIN(sat(1)).PRNTT.Time(2)-PRNdataIN(sat(1)).PRNTT.Time(1));

c = 299792458; %m/s

PRNTT.D_adr = c/F1.*PRNTT.LF1 - c/F2.*PRNTT.LF2;
PRNTT.D_adr = fillmissing(PRNTT.D_adr,"next",'EndValues','previous');
PRNTT.D_adr = fillmissing(PRNTT.D_adr,"next",'EndValues','next');
PRNTT.D_adr_dot = gradient(PRNTT.D_adr);
if all(isnan(PRNTT.D_adr))
    PRNTTOUT = [];
    Slip = [];
else

sigma2 = movvar(PRNTT.D_adr_dot,K);
avg = movmean(PRNTT.D_adr_dot,K);
sigma = sqrt(sigma2);
x=sigmaBounds;

Slip = [];

if bool
figure
subplot(2,1,1)
hold on
plot(PRNTT.Time,PRNTT.D_adr_dot)
plot(PRNTT.Time,avg+x.*sigma,'r')
plot(PRNTT.Time,avg-x.*sigma,'r')
subplot(2,1,2)
hold on
plot(PRNTT.Time,PRNTT.D_adr)
end
for k = 2:length(sigma)
    if abs(PRNTT.D_adr_dot(k) - avg(k)) > x.*sigma(k)
        diff =  PRNTT.D_adr(k-1) - PRNTT.D_adr(k);
        PRNTT.D_adr(k:end) = PRNTT.D_adr(k:end)+diff;
        Slip = [Slip, k];
    end
end
% for s = 1:length(Slip)
%     idx = Slip(s);
% 
%     PRNTT.LF2(idx:end) = F2./c.*(c/F1.*PRNTT.LF1(idx:end)+c/F2.*PRNTT.LF2(idx-1)-c/F1.*PRNTT.LF1(idx-1));
% end
if bool
plot(PRNTT.Time,PRNTT.D_adr)
end


PRNTTOUT = PRNTT;
end
end