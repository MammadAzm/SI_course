clc; clear;

%%

load q3_402123100.mat

N = length(u1);

%%

fprintf("***************************************************************\n")
fprintf("*******System Degree Ident. by Output AutoCorrelation**********\n")
fprintf("---------------------------------------------------------------\n")

Ry1 = [];
Ry2 = [];
for k=1:N
    Ry1(k) = AutoCorrelate(y1, k);
    Ry2(k) = AutoCorrelate(y2, k);
end

queue1 = Ry1(1:50);
queue2 = Ry2(1:50);

order_by_Ry1 = find(queue1==queue1(find(abs(diff(queue1)) == max(abs(diff(queue1))))))+1;
order_by_Ry2 = find(queue2==queue2(find(abs(diff(queue2)) == max(abs(diff(queue2))))))+1;

fprintf(">>> The system degree based on y1 data : %d \n", order_by_Ry1);
fprintf(">>> The system degree based on y2 data : %d \n", order_by_Ry2);

fprintf("---------------------------------------------------------------\n")
fprintf("***************************************************************\n")
fprintf("***************************************************************\n")

