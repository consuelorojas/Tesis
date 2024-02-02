load('mid2_080124.mat');
mid = data(:,1);
fs = 1000;
%lyapExp = lyapunovExponent(mid,fs);
dim = 1;
[~,lag] = phaseSpaceReconstruction(mid,[],dim);
eRange = 200;
lyapExp = lyapunovExponent(mid,fs,2000,dim,'ExpansionRange',eRange)

%lyapMomen = 1/lyapExp;
