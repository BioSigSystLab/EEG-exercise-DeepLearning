clear;
X = load('AccVsTime.mat');
X = exp(X.x);
Y = zeros(7500,1);
for i=1:13;
    s = (i-1)*7500+1;
    e= (i-1)*7500 + 7500;
    A = X(s:e,1);
    Y= Y+A;
end
Y = Y/13;
plot(linspace(1,7500,7500),Y,'LineWidth',2);
hold on;
Y = zeros(7500,1);
for i=14:25;
    s = (i-1)*7500+1;
    e= (i-1)*7500 + 7500;
    A = X(s:e,2);
    Y= Y+A;
end
Y = Y/13;
plot(linspace(1,7500,7500),Y,'LineWidth',2);
ylabel('Confidence');
xlabel('Time Points @2500Hz');
legend('Control','Exercise');