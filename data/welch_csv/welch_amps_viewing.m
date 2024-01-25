filename = 'welch_low0.csv';
M = csvread(filename, 2,1);
figure;
loglog(M(:,1),M(:,2))
hold on
loglog(M(:,1),M(:,3))
hold on
loglog(M(:,1), M(:,4))
hold on
loglog(M(:,1), M(:,5))
grid on
legend('','','','')