%
clear all;
clc;

samePointPixelThreshhold = 1.3;

C_matched = [0 0; 300 300; 3 4; 5 7; 10 11];
C_new = [3.3 4.2; 11.1 9.8; 6 8];

[Locb,distanceToClosestPoint] = dsearchn(C_matched,C_new)
L = distanceToClosestPoint > samePointPixelThreshhold
L = ~L