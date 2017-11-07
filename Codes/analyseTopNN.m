TNN = load('models/TNN.mat');
W1 = TNN.weight1;
B1 = TNN.bias1;
W2 = TNN.weight2;
B2 = TNN.bias2;
W3 = TNN.weight3;
B3 = TNN.bias3;
inp = zeros(1024,1);
out = exp(W3*max(0,W2*max(0,W1*inp+B1)+B2)+B3);
out = out/sum(out);

change = zeros(1024,2);
for i=1:1:1024
   inp = zeros(1024,1);
   inp(i)=10;
   out1 = exp(W3*max(0,W2*max(0,W1*inp+B1)+B2)+B3);
   out1 = out1/sum(out1);
   change(i,1) = out1(1)-out(1);
   change(i,2) = out1(2)-out(2);
end
significant_change = (out(2)-out(1))/2;
figure(1)
plot(min(significant_change*2,change(1:512,1)),'LineWidth',2);
hold on; plot(min(significant_change*2,change(513:1024,1)),'LineWidth',2);