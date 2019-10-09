CON_retention_scores_8h_b5 = [1.644102377,-1.277874885,-6.432599296,-3.083682593,1.852624018,-2.195339873,-3.343612902,-0.491882395,-1.093375708,-3.432207441,-1.275590776,-0.91707894,2.550306001];
EXE_retention_scores_8h_b5 = [3.661700467,5.349308459,-10.14627376,-2.975886839,-5.113791017,-3.844219208,-5.282443917,0.483471026,-3.009656294,-0.705809617,-4.106119571,-1.439179476];
retention_scores_8h_b5 = [CON_retention_scores_8h_b5,EXE_retention_scores_8h_b5];
%figure;
%plot(CON_retention_scores_8h_b5);
%hold on;
%plot(EXE_retention_scores_8h_b5);
CON_retention_scores_24h_8h = [-0.101694619,0.3575334033,0.4012212185,0.6439789717,5.789857737,2.673458102,1.410819818,6.783352486,-3.112821218,4.837110682,1.994754865,3.770115166,2.959655336];
EXE_retention_scores_24h_8h = [3.468671893,0.2445488731,5.068060989,4.177349331,6.452520528,8.07372745,5.95382816,6.612885655,5.78471537,4.566484369,7.987433886,3.706629006];
retention_scores_24h_8h = [CON_retention_scores_24h_8h,EXE_retention_scores_24h_8h];
figure;
plot(CON_retention_scores_24h_8h+CON_retention_scores_8h_b5);
hold on;
plot(EXE_retention_scores_24h_8h+EXE_retention_scores_8h_b5);

A = hdf5read('Layer3_90min.h5','/home/features'); %Layer3.h5 for topographical maps, Layer3_90min.h5 for TF maps
A = permute(A,[2 1]);
A = reshape(A,7500,25,8); %2500 for Topographical Maps, 7500 for TF maps
feats = squeeze(mean(A,1));
feats_CON = feats(1:13,:);
feats_EXE = feats(14:end,:);
feats_with_bias = [ones(25,1),feats];
b = regress(retention_scores_24h_8h',feats_with_bias);
corrScores = zeros(8,1);
corr_pVal = zeros(8,1);
for i=1:8
    [R,p] = corrcoef(feats(:,i),retention_scores_8h_b5);
    corrScores(i) = R(2,1);
    corr_pVal(i) = p(2,1);
end
disp('Step Wise Fit for Retention Scores 24h - 8h');
[B,SE,PVAL,INMODEL,STATS,NEXTSTEP,HISTORY] = stepwisefit(feats,retention_scores_24h_8h');
disp('Step Wise Fit for Retention Scores 8h - Baseline');
[B,SE,PVAL,INMODEL,STATS,NEXTSTEP,HISTORY] = stepwisefit(feats,retention_scores_8h_b5');
disp('Step Wise Fit for Retention Scores 24h - Baseline');
[B,SE,PVAL,INMODEL,STATS,NEXTSTEP,HISTORY] = stepwisefit(feats,(retention_scores_24h_8h+retention_scores_8h_b5)');

figure;
% scatter plot for both CON and EXE
scatter(feats(:,8),retention_scores_24h_8h,200,'filled','HandleVisibility','off'); hold on
% fit a line for all subjects
model= fitlm(feats(:,8),retention_scores_24h_8h); 
P_c = model.Coefficients.Estimate;
x_scal = (min(feats(:,8))-0.05:(max(feats(:,8))-min(feats(:,8))+0.1)/100:max(feats(:,8))+0.05); y_scal = P_c(1) + P_c(2)*x_scal;
plot(x_scal,y_scal,'k--','linewidth',3)
% fit a line for CON subjects
%model= fitlm(feats_CON(:,8),CON_retention_scores_24h_8h); 
%P_c = model.Coefficients.Estimate;
%x_scal = (min(feats_CON(:,8))-0.1:(max(feats_CON(:,8))-min(feats_CON(:,8))+0.2)/100:max(feats_CON(:,8))+0.1); y_scal = P_c(1) + P_c(2)*x_scal;
%plot(x_scal,y_scal,'b--','linewidth',3)
% fit a line for EXE subjects
%model= fitlm(feats_EXE(:,8),EXE_retention_scores_24h_8h); 
%P_c = model.Coefficients.Estimate;
%x_scal = (min(feats_EXE(:,8))-0.1:(max(feats_EXE(:,8))-min(feats_EXE(:,8))+0.2)/100:max(feats_EXE(:,8))+0.1); y_scal = P_c(1) + P_c(2)*x_scal;
%plot(x_scal,y_scal,'r--','linewidth',3)

xlabel('Extracted Feature #8');
ylabel('Retention Motor Score Improvement (24hr - 8hr)');
title('Scatter plot of Retention Motor Scores (24hr - 8hr) vs Extracted feature #8');
legend('Linear fit for all subjects','Linear fit for Control group','Linear fit for Exercise group');