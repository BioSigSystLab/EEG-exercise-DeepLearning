clear; close all;
idx=12;
figure;
data = hdf5read('CON_ccCAM.h5','/home');
data = permute(data,[4 3 2 1]);
%subj1 = mean(data(7500*(idx-1)+1:7500*idx,:,:,:),1);
subj1 = squeeze(mean(data(:,:,:,:),3));
%subj1 = squeeze(data);
subjects = zeros(size(subj1,1)/7500,size(subj1,2));
subjects_timestd = zeros(size(subj1,1)/7500,size(subj1,2)); % for bootstrapping
for i=1:size(subj1,1)/7500
    subjects(i,:) = mean(subj1(7500*(i-1)+1:7500*i,:),1);
    subjects_timestd(i,:) = std(subj1(7500*(i-1)+1:7500*i,:),1)/sqrt(7500); % for bootstrapping
end
subjects = subjects./(subjects_timestd+0.01); % for bootstrapping
con_subjects = subjects;
subj_mean = squeeze(mean(subjects,1));
subj_std = squeeze(std(subjects,1))/sqrt(size(subjects,1));
thresh_level = 0;
subj_mean = interp(subj_mean,2,3);
subj_std = interp(subj_std,2,3);
thresh1 = mean(subj_mean(:))+thresh_level*std(subj_mean(:));
thresh2 = mean(subj_mean(:))-thresh_level*std(subj_mean(:));
subj1_thresh = subj_mean.*((subj_mean>=thresh1)+(subj_mean<thresh2));
%subj1_thresh_freq = squeeze(mean(subj1_thresh,3));
subj1_thresh_freq = subj1_thresh;
%imagesc(squeeze(subj1_thresh_freq));
%errorbar(-interp([subj1_thresh_freq],2),interp([subj_std],2),'LineWidth',2); hold on;
%shadedErrorBar([1:56],-interp([subj1_thresh_freq],2,2),1*interp([subj_std],2,2),'lineprops','b'); hold on;
shadedErrorBar(1:56,-subj1_thresh_freq,subj_std,'lineprops',{'b','HandleVisibility','off'}); hold on;
plot(1:56,-subj1_thresh_freq,'b','DisplayName','CON');

data = hdf5read('EXE_ccCAM.h5','/home');
data = permute(data,[4 3 2 1]);
%subj1 = mean(data(7500*(idx-1)+1:7500*idx,:,:,:),1);
subj1 = squeeze(mean(data(:,:,:,:),3));
%subj1 = squeeze(data);
subjects = zeros(size(subj1,1)/7500,size(subj1,2));
subjects_timestd = zeros(size(subj1,1)/7500,size(subj1,2)); % for bootstrapping
for i=1:size(subj1,1)/7500
    subjects(i,:) = mean(subj1(7500*(i-1)+1:7500*i,:),1);
    subjects_timestd(i,:) = std(subj1(7500*(i-1)+1:7500*i,:),1)/sqrt(7500); % for bootstrapping
end
subjects = subjects./(subjects_timestd+0.01); % for bootstrapping
exe_subjects = subjects;
subj_mean = squeeze(mean(subjects,1));
subj_std = squeeze(std(subjects,1))/sqrt(size(subjects,1));
thresh_level = 0;
subj_mean = interp(subj_mean,2,3);
subj_std = interp(subj_std,2,3);
thresh1 = mean(subj_mean(:))+thresh_level*std(subj_mean(:));
thresh2 = mean(subj_mean(:))-thresh_level*std(subj_mean(:));
subj1_thresh = subj_mean.*((subj_mean>=thresh1)+(subj_mean<thresh2));
%subj1_thresh_freq = squeeze(mean(subj1_thresh,3));
subj1_thresh_freq = subj1_thresh;
%imagesc(squeeze(subj1_thresh_freq));
%errorbar(-interp([subj1_thresh_freq],2),interp([subj_std],2),'LineWidth',2); hold on;
%shadedErrorBar([1:56],-interp([subj1_thresh_freq],2,3),1*interp([subj_std],2,3),'lineprops','r'); hold on;
shadedErrorBar(1:56,-subj1_thresh_freq,subj_std,'lineprops',{'r','HandleVisibility','off'}); hold on;
plot(1:56,-subj1_thresh_freq,'r','DisplayName','EXE');
xlabel('Frequency', 'FontSize',14);
%legend('CON','EXE');
[h,p] = ttest2(mean(con_subjects(:,13:18),2),mean(exe_subjects(:,13:18),2))
% yy = get(gca,'ylim');
% theta_line = line([1 1],[yy(1) yy(2)]);
% delta_line = line([4 4],[yy(1) yy(2)]);
% alpha_line = line([7 7],[yy(1) yy(2)]);
% alpha_end_line = line([11 11],[yy(1) yy(2)]);
% lower_beta_line = line([13 13],[yy(1) yy(2)]);
% higher_beta_line = line([17 17],[yy(1) yy(2)]);
% set([theta_line delta_line alpha_line alpha_end_line lower_beta_line higher_beta_line], 'color', 'k');
% patch([1 1 4 4],[yy(1) yy(2) yy(2) yy(1)],[1 1 1]);
% patch([4 4 7 7],[yy(1) yy(2) yy(2) yy(1)],[1 1 1]);
% patch([7 7 11 11],[yy(1) yy(2) yy(2) yy(1)],[1 1 1]);
% patch([13 13 17 17],[yy(1) yy(2) yy(2) yy(1)],[1 1 1]);
% set(gca,'children',flipud(get(gca,'children')));


% doing stats here
p_vals = zeros(size(con_subjects,2),1);
for i=1:size(con_subjects,2)
%[h,p_vals(i)] = ttest2(con_subjects(:,i),exe_subjects(:,i));
p_vals(i) = anova1([con_subjects(:,i),[exe_subjects(:,i);nan]],'','off');
end
p_vals_interp = interp(p_vals,2,3);
p_vals_interp(p_vals_interp>1) = 1;
figure; plot(p_vals_interp,'--');

% correcting for multiple corrections
win_len = 5;
p_vals_padded = padarray(p_vals_interp,floor(win_len/2),nan,'both');
p_vals_corrected = zeros(size(p_vals,1),1);
for i=1+floor(win_len/2):size(p_vals_padded,1)-floor(win_len/2)
    temp_p_vals = p_vals_padded(i-floor(win_len/2):i+floor(win_len/2));
    temp_p_vals_sorted = sort(temp_p_vals);
    temp_p_vals_corrected = win_len*temp_p_vals_sorted'./linspace(1,win_len,win_len);
    p_vals_corrected(i-floor(win_len/2)) = min(temp_p_vals_corrected);
end
p_vals_corrected(p_vals_corrected>1) = 1;
%p_vals_corrected_interp = interp(p_vals_corrected,2,3);
hold on; plot(p_vals_corrected);
sig_level = 0.06;
x_lims = 1:size(p_vals_corrected,1);
plot(x_lims,sig_level*ones(size(x_lims)),'k--');
ylim([0 1.2]);
xlabel('Frequency', 'FontSize',14);
ylabel('p-values', 'FontSize',14);
lgd = legend('Uncorrected p-values','Simes Corrected p-values','Significance Level');
lgd.FontSize = 14;
