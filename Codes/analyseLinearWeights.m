L1 = load('models/L1.mat');
L2 = load('models/L2.mat');
WL1 = L1.weight;
BL1 = L1.bias;
WL2 = L2.weight;
BL2 = L2.bias;
test = zeros(64,1024);
for j=1:1:1024;
    for i=1:1:64;
        test(i,j) = mean(WL1(j,80*i-79:1:80*i));
    end
end
%plot(abs(test),'LineWidth',3)
surf(max(0.2,abs(test)))
test = zeros(5120,1);
out = max(0,WL1*test + BL1);
plot(out);
out = max(0,WL2*out + BL2);
plot(out);