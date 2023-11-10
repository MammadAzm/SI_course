x = [2.72 4.53 2.75 1.94 4.6 4.23 2.77 1.49 2.56];
y = [0.84 1.28 0.74 1.44 1.39 -0.25 0.05 0.26 0.49];

meanX = 0;
for i=1:length(x)
    meanX = meanX + x(i);
end
meanX = meanX/length(x);

meanY = 0;
for i=1:length(y)
    meanY = meanY + y(i);
end
meanY = meanY/length(y);


varX = 0;
for i=1:length(x)
    temp = (x(i)-meanX)^2;
    varX = varX + temp;
end
varX = varX/length(x);

varY = 0;
for i=1:length(y)
    varY = varY + (y(i)-meanY)^2;
end
varY = varY/length(y);


covXY = sum((x - meanX).*(y - meanY))/(length(x));
