clear; close all;
data1 = hdf5read('CON_ccCAM.h5','/home');
data1 = permute(data1,[4 3 2 1]);
totalPoints1 = size(data1,1);

data2 = hdf5read('EXE_ccCAM.h5','/home');
data2 = permute(data2,[4 3 2 1]);
totalPoints2 = size(data2,1);

thresh_level=0;
%%
%v = VideoWriter('vid_try','MPEG-4');
%{
v = VideoWriter('vid_try','Motion JPEG AVI');
v.FrameRate=10;
open(v);
figure;
set(gcf,'Position',[10,10,560,1120]);
for i=250:5:2500
    subplot(2,1,1);
    %temp1 = squeeze(mean(data1(i:2500:totalPoints1,1,:,:),1));
    temp1 = squeeze(data1(i,1,:,:));
    thresh1 = mean(temp1(:)) + thresh_level*std(temp1(:));
    thresh2 = mean(temp1(:)) - thresh_level*std(temp1(:));
    temp1_thresh = temp1.*((temp1>=thresh1)+(temp1<thresh2));
    imagesc(-temp1_thresh);
    colorbar;
    caxis([-0.2 0.2]);
    viscircles([16.5 16.5],16);
    title(['Control - t = ' num2str(3.0*i/2500)]);
    
    %temp2 = squeeze(mean(data2(i:2500:totalPoints2,1,:,:),1));
    temp2 = squeeze(data2(i,1,:,:));
    thresh1 = mean(temp2(:)) + thresh_level*std(temp2(:));
    thresh2 = mean(temp2(:)) - thresh_level*std(temp2(:));
    temp2_thresh = temp2.*((temp2>=thresh1)+(temp2<thresh2));    
    subplot(2,1,2);
    imagesc(-temp2_thresh);
    colorbar;
    caxis([-0.1 0.1]);
    viscircles([16.5 16.5],16);
    title(['Exercise - t = ' num2str(3.0*i/2500)]);
    frame = getframe(gcf);
    writeVideo(v,frame);
end
close(v)
%}
%%

segments = 1;
for i=1:segments
    if mod(i,5)==1
        figure;
    end
    temp1 = zeros(size(data1,1)/segments,size(data1,2),size(data1,3),size(data1,4));
    temp1_subj = zeros(size(data1,1)/2500,size(data1,2),size(data1,3),size(data1,4)); % for bootstrapping
    temp1_std = zeros(size(data1,1)/2500,size(data1,2),size(data1,3),size(data1,4)); % for bootstrapping
    subjects = size(data1,1)/2500;
    for j=1:subjects
        temp1((2500/segments)*(j-1)+1:(2500/segments)*j,:,:,:) = data1(2500*(j-1)+(2500/segments)*(i-1)+1:2500*(j-1)+(2500/segments)*i,:,:,:);
        temp1_subj(j,:,:,:) = mean(data1(2500*(j-1)+(2500/segments)*(i-1)+1:2500*(j-1)+(2500/segments)*i,:,:,:),1); % for bootstrapping
        temp1_std(j,:,:,:) = std_correct(data1(2500*(j-1)+(2500/segments)*(i-1)+1:2500*(j-1)+(2500/segments)*i,:,:,:),1)/sqrt(2500/segments); % for bootstrapping
    end
    subplot(1,2,1+mod(2*i-2,10));
    temp1 = temp1_subj./temp1_std; % for bootstrapping
    temp1_subj_std = std_correct(temp1,1)/sqrt(subjects); % for bootstrapping
    temp1 = squeeze(mean(temp1,1));%./temp1_subj_std); % for bootstrapping
    thresh1 = mean(temp1(:)) + thresh_level*std(temp1(:));
    thresh2 = mean(temp1(:)) - thresh_level*std(temp1(:));
    temp1_thresh = temp1.*((temp1>=thresh1)+(temp1<thresh2));
    imagesc(-temp1_thresh);
    colorbar;
    caxis([-20 30]);
    viscircles([16.5 16.5],16);
    title(['Control Time avg ' num2str(0.5+(i-1)*3.0/segments) ' to ' num2str(0.5+i*3.0/segments) ' sec'],'FontSize',15);

    temp2 = zeros(size(data2,1)/segments,size(data2,2),size(data2,3),size(data2,4));
    temp2_subj = zeros(size(data2,1)/2500,size(data2,2),size(data2,3),size(data2,4)); % for bootstrapping
    temp2_std = zeros(size(data2,1)/2500,size(data2,2),size(data2,3),size(data2,4)); % for bootstrapping
    subjects = size(data2,1)/2500;
    for j=1:subjects
        temp2((2500/segments)*(j-1)+1:(2500/segments)*j,:,:,:) = data2(2500*(j-1)+(2500/segments)*(i-1)+1:2500*(j-1)+(2500/segments)*i,:,:,:);
        temp2_subj(j,:,:,:) = mean(data2(2500*(j-1)+(2500/segments)*(i-1)+1:2500*(j-1)+(2500/segments)*i,:,:,:),1); % for bootstrapping
        temp2_std(j,:,:,:) = std_correct(data2(2500*(j-1)+(2500/segments)*(i-1)+1:2500*(j-1)+(2500/segments)*i,:,:,:),1)/sqrt(2500/segments); % for bootstrapping
    end
    temp2 = temp2_subj./temp2_std; % for bootstrapping
    temp2_subj_std = std_correct(temp2,1)/sqrt(subjects); % for bootstrapping
    temp2 = squeeze(mean(temp2,1));%./temp2_subj_std); % for bootstrapping
    thresh1 = mean(temp2(:)) + thresh_level*std(temp2(:));
    thresh2 = mean(temp2(:)) - thresh_level*std(temp2(:));
    temp2_thresh = temp2.*((temp2>=thresh1)+(temp2<thresh2));    
    subplot(1,2,1+mod(2*i-1,10));
    imagesc(-temp2_thresh);
    colorbar;
    caxis([-20 30]);
    viscircles([16.5 16.5],16);
    title(['Exercise Time avg ' num2str(0.5+(i-1)*3.0/segments) ' to ' num2str(0.5+i*3.0/segments) ' sec'],'FontSize',15);
end

%doing stats here
con_stats_arr = squeeze(temp1_subj./temp1_std);
exe_stats_arr = squeeze(temp2_subj./temp2_std);
p_vals = zeros(size(con_stats_arr,2),size(con_stats_arr,3));
for i=1:size(p_vals,1)
    for j=1:size(p_vals,2)
        %[h,p_vals(i,j)] = ttest2(con_subjects(:,i,j),exe_subjects(:,i,j));
        p_vals(i,j) = anova1([con_stats_arr(:,i,j),[exe_stats_arr(:,i,j);nan]],'','off');
    end
end
figure; imagesc(p_vals);
viscircles([16.5 16.5],16);
colorbar;
title('Uncorrected p-values','FontSize',15);

% correcting for multiple corrections
win_len = 3;
p_vals_padded = padarray(p_vals,[floor(win_len/2),floor(win_len/2)],nan,'both');
p_vals_corrected = zeros(size(p_vals,1),size(p_vals,2));
for i=1+floor(win_len/2):size(p_vals_padded,1)-floor(win_len/2)
    for j=1+floor(win_len/2):size(p_vals_padded,2)-floor(win_len/2)
        temp_p_vals = p_vals_padded(i-floor(win_len/2):i+floor(win_len/2),j-floor(win_len/2):j+floor(win_len/2));
        temp_p_vals_sorted = sort(temp_p_vals(:));
        temp_p_vals_corrected = win_len*win_len*temp_p_vals_sorted'./linspace(1,win_len*win_len,win_len*win_len);
        p_vals_corrected(i-floor(win_len/2),j-floor(win_len/2)) = min(temp_p_vals_corrected);
    end
end
p_vals_corrected_plot = p_vals_corrected;
p_thresh = 0.06;
%p_vals_corrected_plot(p_vals_corrected>p_thresh)=1;%-p_thresh;
p_vals_corrected_plot(p_vals_corrected<1e-4)=1;%-p_thresh;
figure; subplot(1,2,1); imagesc(-log(p_vals_corrected_plot));
viscircles([16.5 16.5],16);
colorbar;
title('Simes corrected p-values (-log(p) plotted)','FontSize',15);

p_vals_corrected_thresh = p_vals_corrected_plot;
p_vals_corrected_thresh(p_vals_corrected_thresh>p_thresh)=1;%-p_thresh;
subplot(1,2,2); imagesc(-log(p_vals_corrected_thresh));
viscircles([16.5 16.5],16);
colorbar;
title('Simes corrected p-values thresholded at 0.05 (-log(p) plotted)','FontSize',15);

function corrected_std = std_correct(x,axis)
    corrected_std = std(x,axis);
    corrected_std(find(corrected_std<=0.01)) = corrected_std(find(corrected_std<=0.01))+100;
end
%%
%{
j=1;
time_array = [1495,1510,1530];
for i=1495:20:1585
    subplot(2,5,j);
    %i = time_array(t);
    %temp1 = squeeze(mean(data1(i:2500:totalPoints1,1,:,:),1));
    temp1 = squeeze(data1(i,1,:,:));
    thresh1 = mean(temp1(:)) + thresh_level*std(temp1(:));
    thresh2 = mean(temp1(:)) - thresh_level*std(temp1(:));
    temp1_thresh = temp1.*((temp1>=thresh1)+(temp1<thresh2));
    imagesc(-temp1_thresh);
    %colorbar;
    caxis([-0.15 0.15]);
    viscircles([16.5 16.5],16);
    title(['Control - t = ' num2str(3.0*i/2500)]);
    
    %temp2 = squeeze(mean(data2(i:2500:totalPoints2,1,:,:),1));
    temp2 = squeeze(data2(i,1,:,:));
    thresh1 = mean(temp2(:)) + thresh_level*std(temp2(:));
    thresh2 = mean(temp2(:)) - thresh_level*std(temp2(:));
    temp2_thresh = temp2.*((temp2>=thresh1)+(temp2<thresh2));    
    subplot(2,5,j+5);
    imagesc(-temp2_thresh);
    %colorbar;
    caxis([-0.15 0.15]);
    viscircles([16.5 16.5],16);
    title(['Exercise - t = ' num2str(3.0*i/2500)]);
    j=j+1;
end
%}