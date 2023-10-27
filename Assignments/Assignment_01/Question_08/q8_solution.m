% clc; clear;

k = 100 ;

sys = tf([k -k],[1 2 1 0]);

bode(sys)

grid on