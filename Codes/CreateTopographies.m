
%clear
load('Good_Pos.mat') % position of the electrodes, taken from brainstorm

%arna
% data vector, just for illustration

%v=[6.64530389547269e-13;6.63368201961991e-13;1.14267071484424e-12;6.62089558331765e-13;7.66622519399818e-13;6.78278117741237e-13;8.43098582382908e-13;5.96430517075009e-13;5.05766466760381e-13;4.75115798111346e-13;4.77263989150664e-13;1.15978427320815e-12;3.64535853831862e-13;4.75885131898716e-13;3.73472632131793e-13;9.85742026883893e-13;1.11837366300129e-12;5.04504582375209e-13;3.97638969542097e-13;3.90070153002426e-13;4.96028757233680e-13;7.88041108010289e-13;9.71187396655674e-13;6.63086085638494e-13;6.65088814626588e-13;6.43921143582671e-13;1.38393435514223e-12;7.09851088857296e-13;9.48200612691247e-13;9.58427540185136e-13;9.78380746033772e-13;9.39927488194114e-13;8.55535467199260e-13;6.99525139307451e-13;7.39957671355027e-13;7.49462952465397e-13;7.97058332035157e-13;7.31551169714624e-13;7.20264556769500e-13;6.64005083552514e-13;1.82886627702704e-12;1.18216837157425e-12;4.22658014563502e-13;3.86739286470683e-13;9.21292216026933e-13;1.39129009728842e-12;4.64516586367828e-13;3.69479299292968e-13;3.64301682183623e-13;4.67596814214354e-13;9.51068407278049e-13;4.17725343410139e-13;4.48402610838008e-13;4.03876101490940e-13;7.94448705570191e-13;8.79530201589366e-13;5.79839450049741e-13;5.90401827055377e-13;8.70909587209906e-13;1.12971921362786e-12;1.09862128057629e-12;1.04233869894715e-12;1.09993060348661e-12;1.07862888015462e-12] ;
files = dir('*EXE*90*.mat');
filenames = extractfield(files,'name');
for f=1:size(files,1)
    topographies = zeros(2500,3,64,64);
    char(filenames(f));
    load(char(filenames(f)));
       
    % compilation of the electrode locations
    for p=1:64
        x(p,1) = Good_Pos.(RowNames{p})(1) ;
        y(p,1) = Good_Pos.(RowNames{p})(2) ;
    end
    
    im_scale = 63/(max(x)-min(x));
    x = im_scale*(x-min(x));
    y = im_scale*(y-min(y));
        
    save_str = ['Topographies/top_',char(filenames(f))];
    
    tic
    for i=2501:3:10000
        for j=0:2
            v = mean(TF(:,i+j,23:33),3);
            
            % create the coordinates vector for interpolation
            xq = repmat(round(min(x)):1:round(max(x)),64,1) ;
            yq = repmat(round(min(y))-6:1:round(max(y))+6,64,1)' ;
            
            % Interpolation of the data
            vq = griddata(x,y,v,xq,yq,'cubic') ; %hold on
            vq(isnan(vq)) = 0;
            %vq = imresize(vq,[100,100]);
            %vq = vq/std(vq(:));
            topographies(round((i-2501)/3)+1,j+1,:,:) = vq;
            % display the interpolated data
            %imagesc(xq(:,1),yq(1,:),vq)
            
            % display the position of the electrodes
            %for p=1:64
            %    plot(Good_Pos.(E_names{p})(1),Good_Pos.(E_names{p})(2),'o') ; hold on
            %end
            
            %axis equal % just because it is more beautiful
            %axis off % same reason
            %axis ij % revert the y axis because the electrodes were flipped up and down
        end
    end
    toc
    save(save_str,'topographies','-v7.3')
end