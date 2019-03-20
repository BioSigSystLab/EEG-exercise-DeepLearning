clear;
files = dir('*Matching60*.mat');
filenames = extractfield(files,'name');
tic
tf_tensor = zeros(size(files,1),7500,64,55);
for f=1:size(files,1)
    temp_tf = load(char(filenames(f)));
    temp_tf = permute(temp_tf.TF(:,2501:10000,:),[2 1 3]);
    tf_tensor(f,:,:,:,:) = temp_tf;
end
hdf5write('60min_h5.h5','/home',tf_tensor);
toc