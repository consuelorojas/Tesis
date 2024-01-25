filename = 'welch_amplitudes.csv';
filename2 = 'welch_amplitudes_high.csv'
M2 = csvread(filename2,1,1);
M = csvread(filename, 1,1);
x = M2(:,2).*M2(:,1).^(5/2);
figure;
loglog(M2)
hold on
loglog(x)
grid on

