function PRNdata = BiasAdjust(PRNdata,sat)
    for j = 1:length(sat)
        i = sat(j);
        PRNdata(i).PRNTT.sTEC = -1*1e-16*(PRNdata(i).F1.^2*PRNdata(i).F2.^2/((PRNdata(i).F2.^2 - PRNdata(i).F1.^2) *40.3)).*(PRNdata(i).PRNTT.PseudoDiff_cor-PRNdata(i).Bias);
        PRNdata(i).PRNTT.sTEC_noBias = -1*1e-16*(PRNdata(i).F1.^2*PRNdata(i).F2.^2/((PRNdata(i).F2.^2 - PRNdata(i).F1.^2) *40.3)).*(PRNdata(i).PRNTT.PseudoDiff_cor);
    end

end