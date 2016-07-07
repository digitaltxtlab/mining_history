wd = 'C:\Users\KLN\Documents\textAnalysis\bible\newTestament';
ddFull = 'C:\Users\KLN\Documents\textAnalysis\bible\newTestament\ntTotal';
cd(wd);
%%
[to, vocabfull, docstokens, filename] = tdmVanilla(ddFull);
% Heaps' law
N = numel(vocabfull);
n = zeros(size(docstokens,1),2); 
for i = 1:length(n);
    n(i,1) = length(docstokens{i});
    n(i,2) = length(unique(docstokens{i})); 
end;
figure(1)
subplot(221),h1 = plot(sort(n(:,2))./N,'k');
[~,idx] = sort(n(:,1));
sdt = docstokens(idx);
sdtcol = {};
heapsf = zeros(size(sdt));
for ii = 1:numel(sdt)
    if ii == 1; 
        sdtcol{ii} = sdt{ii};  %#ok<SAGROW>
    else    
        sdtcol{ii} = [sdtcol{ii-1} sdt{ii}]; %#ok<SAGROW>
    end
    heapsf(ii) = length(unique(sdtcol{ii}))/N;
end
figure(1)
subplot(222),h2 = plot(heapsf,'k');
% Zipf's law
zipf = sort(sum(to),'descend')'; 
figure(1); subplot(223), %h3 = scatter(log(1:numel(zipf)),log(zipf),'k.'); 
 h3 = scatter(1:numel(zipf),zipf,'k.'); 
set(gca,'xscale','log','yscale','log')
box on
figure(1); subplot(224), h4 = loglog(zipf,'k'); 
plotCorrect
%%
filename = regexprep(filename,'\.txt','');
metaclass = {'paul','nonpaul','nonpaul','paul','paul','paul','nonpaul', ...
    'nonpaul','paul','paul','nonpaul','history','paul','paul','paul', ...
    'nonpaul','nonpaul','history','nonpaul','history','history', ...
    'history','paul','paul','nonpaul','paul','paul'};
segsize = 100;
docn = size(docstokens,1);
[segcor,segclass] = deal({});
iii = 1;
for i = 1:docn
    temp = docstokens{i};
    n = (length(temp) - segsize) + 1;
    for ii = 1:n    
        segcor{iii} = temp(ii:ii+segsize-1);
        segclass(iii,1) = filename(i);
        segclass(iii,2) = metaclass(i);
        iii = iii+1;
    end
end
segcor = segcor';
%%

fileN = length(segcor);
[vocab,void,index,frequencies] = deal(cell(fileN,1));
[words,vocabwords] = deal(zeros(fileN,1));
for i = 1:fileN
    disp(i)
    words(i) = length(segcor{i});
    [vocab{i},void{i},index{i}] = unique(segcor{i});
    vocabwords(i) = length(vocab{i});
    frequencies{i} = hist(index{i},vocabwords(i));
end
vocabfull = unique([vocab{:}]);
vocabfullN = length(vocabfull);
to = zeros(fileN,vocabfullN);
for i = 1:fileN
    disp(['dtm for doc' ' ' num2str(i)])
    for ii = 1:vocabfullN
        to(i,ii) = sum(strcmp(vocabfull(ii),segcor{i}));
    end
end

to100Chunk = to;

% save('to100Chunk.mat','-v7.3')
%% chunks with unique content for each book
wd = 'C:\Users\KLN\Documents\textAnalysis\bible\newTestament';
ddFull = 'C:\Users\KLN\Documents\textAnalysis\bible\newTestament\ntTotal';
cd(wd);
[to, vocabfull, docstokens, filename] = tdmVanilla(ddFull);
filename = regexprep(filename,'\.txt','');
metaclass = {'paul','nonpaul','nonpaul','paul','paul','paul','nonpaul', ...
    'nonpaul','paul','paul','nonpaul','history','paul','paul','paul', ...
    'nonpaul','nonpaul','history','nonpaul','history','history', ...
    'history','paul','paul','nonpaul','paul','paul'};
% document word length
n = zeros(size(docstokens));
for i = 1:length(docstokens); n(i) = length(docstokens{i}); end;
% chunk size
csize = 100;
% chunked docstokens
lencdt = 0; % total number of chunks
[cdt, cdtcol] = deal({});% chunks and all chucks in one cell of cells
cclass = {};% class labels for chunks
for i = 1:length(docstokens) 
    doc = docstokens{i};
    len = length(doc);
    cs = 1:csize:len;
    for ii = 1:length(cs)
        if ii == length(cs)
           cdt{i}{ii} = doc(cs(ii):end);
        else
           cdt{i}{ii} = doc(cs(ii):(cs(ii) + (csize-1)));
        end  
    end
    lencdt = lencdt + length(cdt{i});
    cdtcol = [cdtcol cdt{i}];
    cclass = [cclass; repmat(metaclass(i),length(cdt{i}),1)];
end
cdtcol = cdtcol';
% make chunk-term matrix from cdtcol 
[cvocab,void,index,cfrequencies] = deal(cell(length(cdtcol),1));
[cwords,cvocabwords] = deal(zeros(length(cdtcol),1));
for i = 1:length(cdtcol)
    cwords(i) = length(cdtcol{i});
    [cvocab{i},void{i},index{i}] = unique(cdtcol{i});
    cvocabwords(i) = length(cvocab{i});
    cfrequencies{i} = hist(index{i},cvocabwords(i));
end
cvocabfull = unique([cvocab{:}]);
cvocabfullN = length(cvocabfull);
ctm = zeros(length(cdtcol),cvocabfullN);
for i = 1:length(cdtcol)
    for ii = 1:cvocabfullN
        ctm(i,ii) = sum(strcmp(cvocabfull(ii),cdtcol{i}));
    end
end
% reduce sparsity chunk-term matrix
spt = .95; % allowed sparsity of a word
nchunk = round(size(ctm,1)*spt);
rctm = [];% reduced chunk-term matrix
rvocab = {};
for k = 1:size(ctm,2)
    %k = 1
    if sum(ctm(:,k) == 0) <= nchunk 
        rctm = [rctm ctm(:,k)];
        rvocab = [rvocab cvocabfull(k)];
    end
end
size(rctm)
rvocab
% save('ntFeatures.mat','cclass','ctm','cvocabfull','nchunk','','rctm','rvocab','spt')
%% quick and dirty multi-layered perceptron for reduced chunk term matrix
load('ntFeatures.mat')
per = unique(cclass);% classes
c = size(per,1);
m = size(rctm,1); n = size(rctm,2);
% full set
inputs = rctm';
targets = zeros(m,c);
for i = 1:c
    targets(strncmp(per{i},cclass,length(per{i})),i) = 1;
end
targets = targets';
subplot(121),
imagesc(targets); colormap('bone'); colormap(flipud(colormap));
xlabel('documents')
subplot(122),
imagesc(inputs);xlabel('documents');ylabel('features');
% create network
tic
hiddenLayerSize = 8;
net = patternnet(hiddenLayerSize);
% set up division of data
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio   = 15/100;
net.divideParam.testRatio  = 15/100;
% suppress gui
net.trainParam.showWindow = false;
% train network
rng(0) 
[net,tr] = train(net,inputs,targets);
toc
% performance on training set
outputs = net(inputs);% compute network output
errors = gsubtract(targets,outputs);% compute error
figure, plotperform(tr)
% plot confusion matrix
h = plotconfusion(targets,outputs);
    set(gca,'xticklabel',per,'yticklabel',per)
    plotCorrect
        %figName = strjoin({direct '\\annConfusion.tiff'},'');
        %saveFig(figName);
% confusion matrix from figure 5
hf = figure(5);   
set(hf,'Color',[1 1 1],'Name','Confusion matrix')
set(hf,'Position', [100, 100, 1049*.45, 895*.45])
cmat = [1029 48 26 41 170 22 15 78 393];
c2mat = reshape(zeroOneScale(cmat),3,3);
cmat = reshape(cmat,3,3);
imagesc(c2mat)
colormap(flipud(gray));
for i = 1:3
    for ii = 1:3
        ht = text(i-(length(num2str(cmat(i,ii))))/20,ii,num2str(cmat(i,ii)));
        ht.FontSize = 14;
        ht.FontWeight = 'bold';
        ht.FontName = 'Times';
        if (i == 1 & ii == 1)|(i == 3 & ii == 3)
            ht.Color = [1 1 1];
        end
    end
end
classname = {'Historical','Non-Pauline','Pauline'};
set(gca,'xtick',1:3,'ytick',1:3,'xticklabel',classname,...
    'yticklabel',classname,'yticklabelrotation',25 ...
    ,'xticklabelrotation',25)
v = vline([1.5 2.5],'k-');[v(1).LineWidth,v(2).LineWidth] = deal(2);
h = hline([1.5 2.5],'k-');[h(1).LineWidth,h(2).LineWidth] = deal(2);
plotCorrect
xlabel('Target','FontSize',18); ylabel('Output','FontSize',18)
% saveFig(strcat(wd,'\confusionmat.tiff'))
%
c = confusion(targets,outputs);
baselines = sum(targets,2)./sum(sum(targets,2));
baseline = baselines(1);

% estimate performance
actualClassMat = targets';
% build prediction matrix by selecting max class
predictedClassMat = outputs';
for k = 1:size(predictedClassMat,1)
    predictedClassMat(k,:) = double(predictedClassMat(k,:) == max(predictedClassMat(k,:)));
end
[actualClassVec, predictedClassVec] = deal(zeros(size(actualClassMat,1),1));
[actualClassStr, predictedClassStr] = deal(cell(size(actualClassMat,1),1));
for i = 1:length(per)
    ii = find(actualClassMat(:,i) == 1);
    iii = find(predictedClassMat(:,i) == 1);
    actualClassVec(ii,1) = i;
    predictedClassVec(iii,1) = i;
    actualClassStr(ii,1) = per(i);
    predictedClassStr(iii,1) = per(i);
end
perfEval = Evaluate(actualClassVec,predictedClassVec);
figure(3)
barh(perfEval,'k')
set(gca,'yticklabel',{'accuracy' 'sensitivity' 'specificity' 'precision' 'recall' 'F' 'G'})
box off
plotCorrect
cp = classperf(actualClassStr, predictedClassStr); cp
%% redo classifier with Gospel of Thomas as unknown data
ntAlternate = 'C:\Users\KLN\Documents\textAnalysis\bible\newTestament\ntAlternate';
% tokenize
[to, vocabfull, docstokens, filename] = tdmVanilla(ntAlternate);
filename = regexprep(filename,'\.txt','');
metaclass = {'paul','nonpaul','nonpaul','paul','paul','paul','nonpaul', ...
    'nonpaul','paul','paul','nonpaul','history','paul','paul','paul', ...
    'nonpaul','nonpaul','history','nonpaul','history','history', ...
    'history','paul','paul','nonpaul','paul','thomas','paul'};
% document word length
n = zeros(size(docstokens));
for i = 1:length(docstokens); n(i) = length(docstokens{i}); end;
% chunk size
csize = 100;
% chunked docstokens
lencdt = 0; % total number of chunks
[cdt, cdtcol] = deal({});% chunks and all chucks in one cell of cells
cclass = {};% class labels for chunks
for i = 1:length(docstokens) 
    doc = docstokens{i};
    len = length(doc);
    cs = 1:csize:len;
    for ii = 1:length(cs)
        if ii == length(cs)
           cdt{i}{ii} = doc(cs(ii):end);
        else
           cdt{i}{ii} = doc(cs(ii):(cs(ii) + (csize-1)));
        end  
    end
    lencdt = lencdt + length(cdt{i});
    cdtcol = [cdtcol cdt{i}];
    cclass = [cclass; repmat(metaclass(i),length(cdt{i}),1)];
end
cdtcol = cdtcol';
% make chunk-term matrix from cdtcol 
[cvocab,void,index,cfrequencies] = deal(cell(length(cdtcol),1));
[cwords,cvocabwords] = deal(zeros(length(cdtcol),1));
for i = 1:length(cdtcol)
    cwords(i) = length(cdtcol{i});
    [cvocab{i},void{i},index{i}] = unique(cdtcol{i});
    cvocabwords(i) = length(cvocab{i});
    cfrequencies{i} = hist(index{i},cvocabwords(i));
end
cvocabfull = unique([cvocab{:}]);
cvocabfullN = length(cvocabfull);
ctm = zeros(length(cdtcol),cvocabfullN);
for i = 1:length(cdtcol)
    for ii = 1:cvocabfullN
        ctm(i,ii) = sum(strcmp(cvocabfull(ii),cdtcol{i}));
    end
end
% reduce sparsity chunk-term matrix
spt = .95; % allowed sparsity of a word
nchunk = round(size(ctm,1)*spt);
rctm = [];% reduced chunk-term matrix
rvocab = {};
for k = 1:size(ctm,2)
    %k = 1
    if sum(ctm(:,k) == 0) <= nchunk 
        rctm = [rctm ctm(:,k)];
        rvocab = [rvocab cvocabfull(k)];
    end
end
% extract two subcorpora NT and Thomas
it = strmatch('thomas',cclass);
trctm = rctm(it,:);
rctm = rctm([1:(it(1)-1) (it(end)+1):end],:);
cclass = cclass([1:(it(1)-1) (it(end)+1):end],:); % remove thomas class
% build ANN and train classifier
per = unique(cclass);% classes
c = size(per,1);
m = size(rctm,1); n = size(rctm,2);
% full set
inputs = rctm';
targets = zeros(m,c);
for i = 1:c
    targets(strncmp(per{i},cclass,length(per{i})),i) = 1;
end
targets = targets';
subplot(121),
imagesc(targets); colormap('bone'); colormap(flipud(colormap));
xlabel('documents')
subplot(122),
imagesc(inputs);xlabel('documents');ylabel('features');
% create network
hiddenLayerSize = 8;
net = patternnet(hiddenLayerSize);
% set up division of data
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio   = 15/100;
net.divideParam.testRatio  = 15/100;
% suppress gui
net.trainParam.showWindow = false;
% train network
rng(0) 
[net,tr] = train(net,inputs,targets);
% performance on training set
outputs = net(inputs);% compute network output
errors = gsubtract(targets,outputs);% compute error
figure, plotperform(tr)
% plot confusion matrix
h = plotconfusion(targets,outputs);
    set(gca,'xticklabel',per,'yticklabel',per)
    plotCorrect
% confusion matrix from figure 5
hf = figure(5);   
set(hf,'Color',[1 1 1],'Name','Confusion matrix')
set(hf,'Position', [100, 100, 1049*.45, 895*.45])
cmat = [1052 42 20 25 195 28 8 59 393];
c2mat = reshape(zeroOneScale(cmat),3,3);
cmat = reshape(cmat,3,3);
imagesc(c2mat)
colormap(flipud(gray));
for i = 1:3
    for ii = 1:3
        ht = text(i-(length(num2str(cmat(i,ii))))/20,ii,num2str(cmat(i,ii)));
        ht.FontSize = 14;
        ht.FontWeight = 'bold';
        ht.FontName = 'Times';
        if (i == 1 & ii == 1)|(i == 3 & ii == 3)
            ht.Color = [1 1 1];
        end
    end
end
classname = {'Historical','Non-Pauline','Pauline'};
set(gca,'xtick',1:3,'ytick',1:3,'xticklabel',classname,...
    'yticklabel',classname,'yticklabelrotation',25 ...
    ,'xticklabelrotation',25)
v = vline([1.5 2.5],'k-');[v(1).LineWidth,v(2).LineWidth] = deal(2);
h = hline([1.5 2.5],'k-');[h(1).LineWidth,h(2).LineWidth] = deal(2);
plotCorrect
xlabel('Target','FontSize',18); ylabel('Output','FontSize',18)
%saveFig(strcat(wd,'\confusionmat.tiff'))
%
c = confusion(targets,outputs);
baselines = sum(targets,2)./sum(sum(targets,2));
baseline = baselines(1);
% estimate performance
actualClassMat = targets';
% build prediction matrix by selecting max class
predictedClassMat = outputs';
for k = 1:size(predictedClassMat,1)
    predictedClassMat(k,:) = double(predictedClassMat(k,:) == max(predictedClassMat(k,:)));
end
[actualClassVec, predictedClassVec] = deal(zeros(size(actualClassMat,1),1));
[actualClassStr, predictedClassStr] = deal(cell(size(actualClassMat,1),1));
for i = 1:length(per)
    ii = find(actualClassMat(:,i) == 1);
    iii = find(predictedClassMat(:,i) == 1);
    actualClassVec(ii,1) = i;
    predictedClassVec(iii,1) = i;
    actualClassStr(ii,1) = per(i);
    predictedClassStr(iii,1) = per(i);
end
perfEval = Evaluate(actualClassVec,predictedClassVec);
figure(3)
barh(perfEval,'k')
set(gca,'yticklabel',{'accuracy' 'sensitivity' 'specificity' 'precision' 'recall' 'F' 'G'})
box off
plotCorrect
cp = classperf(actualClassStr, predictedClassStr); cp
% appy clssifier to Thomas and plot result
tinputs = trctm';
toutputs = net(tinputs);
% build prediction matrix by selecting max class
predictedClassMat = toutputs';
for k = 1:size(predictedClassMat,1)
    predictedClassMat(k,:) = double(predictedClassMat(k,:) == max(predictedClassMat(k,:)));
end
figure(1)
h = bar(sum(predictedClassMat));
h.FaceColor = [0 0 0];
set(gca,'xticklabel',per)
ylim([0 60]); xlabel('Predicted class'); ylabel('Number of slices')
col12(1); plotCorrect
