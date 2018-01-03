logs = dir('*');
isub = [logs(:).isdir]&~ismember({logs(:).name},{'.','..'});
logs = logs(isub);

for i=1:length(logs)
    csvPath = fullfile(logs(i).name,'log.csv');
    
    data = csvread(csvPath,1,0);
    fid = fopen(csvPath,'r');
    headers = fgetl(fid);
    fclose(fid);
    headers = split(headers,',');

    logs(i).data = data;
    logs(i).headers = headers;
    
    h = findobj('Type','Figure','Name',logs(i).name);
    if isempty(h) 
        F = figure('Name',logs(i).name);
    else
        figure(h);
    end
    ix = strcmp(headers,'_Episode');
    iy = strcmp(headers,'KL');
    plot(data(:,ix),data(:,iy));
end
return;
%% PLOT 2 VS 3 VS 6 DOF rewards
h = findobj('Type','Figure','Name','2v3v6');
if isempty(h)
    F = figure('Name','2v3v6');
else
    figure(h);
end
idx = [8,7,5];
maxrew = [500*100,500*100,1000*0];
minrew = [500*0.1059,500*0.1589,1000*-1];
for i = 1:3
    x = logs(idx(i)).data(:,strcmp(logs(idx(i)).headers,'_Episode'));
    y = logs(idx(i)).data(:,strcmp(logs(idx(i)).headers,'_MeanReward'));
    % % Scale y to be between 0 and 1
    ybar = mean(y);
    ystd = std(y);
    ymin = minrew(i);
    ymax = maxrew(i);
    y = (y-ybar)/ystd;
    ymin = (ymin-ybar)/ystd;
    ymax = (ymax-ybar)/ystd;
    y = rescale(y,'inputmin',ymin,'inputmax',ymax);
    plot(x,y);
    hold on;
end
hold off;
xlabel('Episode');
ylabel('Reward %');
legend("f(\phi,\theta)","f(\phi,\theta,\psi)","f(x,y,z,\phi,\theta,\psi,t)",'location','northwest');
ax = gca;
set(ax,'fontname','times new roman','fontsize',12);
axis 'tight';
ylim([0,0.145]);
%% PLOT 200 VS 500 VS 1000
h = findobj('Type','Figure','Name','200v500v1000');
if isempty(h)
    F = figure('Name','200v500v1000');
else
    figure(h);
end
idx = [10,8,4];
multiple = [200,500,1000];
maxrew = [200*100,500*100,1000*100];
minrew = [200*0.1059,500*0.1059,1000*0.1059];
for i = 1:3
    x = logs(idx(i)).data(:,strcmp(logs(idx(i)).headers,'_Episode'));
    x = [0;x];
    x = x*multiple(i);
    y = logs(idx(i)).data(:,strcmp(logs(idx(i)).headers,'_MeanReward'));
    % % Scale y to be between 0 and 1
    ybar = mean(y);
    ystd = std(y);
    ymin = minrew(i);
    ymax = maxrew(i);
    y = (y-ybar)/ystd;
    ymin = (ymin-ybar)/ystd;
    ymax = (ymax-ybar)/ystd;
    y = rescale(y,'inputmin',ymin,'inputmax',ymax);
    y = [0;y];
    plot(x,y);
    hold on;
end
hold off;
xlabel('Steps');
ylabel('Reward %');
axis 'tight';
xlim([0,6e6]);
ylim([0,0.14]);
T = text(2.6e6,0.02,'Early Termination');
ax = gca;
set(ax,'fontname','times new roman','fontsize',12);
set(T,'fontname','times new roman','fontsize',12);
arrow([2.6e6,0.02],[2.35e6,0.02],'length',10);
legend("200 Steps/Episode","500 Steps/Episode","1000 Steps/Episode",'location','northwest');
