clear;
clc;
%% img_1
rowmin1 = 0 ; %y
colmin1 = 0 ; %x
rowmax1 = 758;%y
colmax1 = 530;%x
M1 = [] ;
%% img_2
rowmin11 = 51 ; %y
colmin11 = 283; %x
rowmax11 = 809; %y
colmax11 = 813; %x
M2 = [] ;
%% 计算
x2 = colmin11 ;
y2 = rowmin11 ;
for x1 = colmin1:colmax1
    for y1 = rowmin1:rowmax1
        
        y2 = y2+1;
    end
    x2 = x2+1 ;
end