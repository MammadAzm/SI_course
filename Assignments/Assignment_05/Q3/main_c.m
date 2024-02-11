clc; clear;

%%

load q3_402123100.mat

N = length(u1);

%%

fprintf("***************************************************************\n")
fprintf("******System Delay Ident. by Input-Output CrossCorrelation*****\n")
fprintf("---------------------------------------------------------------\n")

Ruy1 = [];
Ruy2 = [];
for k=1:N
    Ruy1(k) = CrossCorrelate(u1,y1,k);
    Ruy2(k) = CrossCorrelate(u2,y2,k);
end

queue1 = Ruy1(1:50);
queue2 = Ruy2(1:50);

delay_by_Ruy1 = find(queue1==queue1(find(abs(diff(queue1)) == max(abs(diff(queue1))))))+1;
delay_by_Ruy2 = find(queue2==queue2(find(abs(diff(queue2)) == max(abs(diff(queue2))))))+1;

fprintf(">>> The system delay based on y1 data : %d \n", delay_by_Ruy1);
fprintf(">>> The system delay based on y2 data : %d \n", delay_by_Ruy2);

fprintf("---------------------------------------------------------------\n")
fprintf("***************************************************************\n")
fprintf("***************************************************************\n")

