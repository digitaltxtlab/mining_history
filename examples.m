kill
wd = 'C:\Users\KLN\Documents\textAnalysis\bible\newTestament';
ddFull = 'C:\Users\KLN\Documents\textAnalysis\bible\newTestament\ntTotal';

cd(wd)
%% 
% make collected NT
ddFull = 'C:\Users\KLN\Documents\textAnalysis\bible\newTestament\ntTotal';
cd(ddFull)
filename = getAllFiles(ddFull);
% make empty file
fid = fopen( 'ntMain.txt', 'wt' );
fclose(fid);
% make copy command for dos
strmain = 'copy ntMain.txt';
for i = 1:length(filename);
str = filename(i);
e = regexp(str,'\\');e  = cell2mat(e); e = e(end)+1;
filename{i}= str{1}(e:end);
strmain = strcat(strmain,'+',filename{i});
end
system(strmain)
%
filename = getAllFiles(ddFull);
doc = fileread(filename{end});
% delete collected NT file
delete('ntMain.txt')
% preprocess NT
temp = strtrim(regexprep(doc,'([^\D]+)',' '));% remove numbers
temp = strtrim(regexprep(temp,'([^\w]+)',' '));% non-alphabetic characters
temp = regexprep(['''',temp,''''],' ',''','''); % insert '' and , around alphabetic sequence
evalc(['tokens = {',temp,'}']); % assign tokens
doctokens = lower(tokens);
% extract word frequencies
words = length(doctokens);
[vocab,void,index] = unique(doctokens);
vocabwords = length(vocab);
frequencies = hist(index,vocabwords);
[ranked_frequencies,idx] = sort(frequencies,'descend');
ranked_vocabulary = vocab(idx);
%% make heaps' distribution
heapslaw = zeros(words,2);
for k = 1:words
    if mod(k,1000) == 0;
        disp(k);
    end
    heapslaw(k,1) = k;    
    heapslaw(k,2) = length(unique(doctokens(1:k)));
end
save('heapDist.mat','heapslaw')
%% Burstiness, Zipf and Heaps for figures 1

% ### Zipf
hf = figure(1);

set(hf,'Color',[1 1 1],'Name','Burstiness of NT')
set(hf,'Position', [100, 100, 1049*.5, 895*.5])
subplot(223)
h1 = scatter(1:vocabwords,ranked_frequencies);
h1.Marker = '.'; h1.MarkerEdgeColor = [0 0 0];
set(gca,'xscale','log','yscale','log')
limits = axis; axis([1,vocabwords,limits(3:4)])
xlabel('Word rank')
ylabel('Word Frequency')
% ### Heaps
cd(wd)
load('heapDist.mat')
subplot(224),
h3 = scatter(heapslaw(:,1),heapslaw(:,2));
h3.Marker = '.'; h3.MarkerEdgeColor = [0 0 0];
h3.SizeData = 6;
xlabel('Collection size')
ylabel('Vocabulary size')
set(gca,'xticklabel',{'0' '1000' '2000'})
% ### burstiness
%
% for single word (publication)
subplot(221)
aword = 'jesus';
intervals = diff(find(strcmp(doctokens,aword)));
nvals = length(intervals);
h1 = plot(1:nvals,intervals);
h1.Color = [.5 .5 .5]; 
h2 = hline(mean(intervals));
h2.LineWidth = 2; h2.LineStyle = ':'; h2.Color = [0 0 0];
limits = axis; axis([1,nvals,0,4000])
xlabel('Consecutive occurrences')
ylabel('Word distance')
box off
subplot(222)
h4 = histogram(intervals);
h4.NumBins = 100;
h4.BinWidth = 100;
h4.EdgeColor = [1 1 1];
h4.FaceColor = [0 0 0];
xlim([-100 3100]); ylim([-10 600])
xlabel('Word distance')
ylabel('Frequency')
%h5 = vline(75);
%h5.LineWidth = 2; h5.LineStyle = ':'; h5.Color = [0 0 0];
box off
plotCorrect
%saveFig(strcat(wd,'\figure1.tiff'))
%% for multiple words
twowords = {'jesus' 'wine'};
%%retrieve ranking positions
find(strcmp(ranked_vocabulary,twowords{1}))
find(strcmp(ranked_vocabulary,twowords{2}))
hf = figure(4);
set(hf,'Color',[1 1 1],'Name','Burstiness of NT')
% generate the plots
for k = 1:2
    %k = 1
    intervals = diff(find(strcmp(doctokens,twowords{k})));
    subplot(2,1,k), nvals = length(intervals);
    plot(1:nvals,intervals,'-r',[1,nvals], ones(1,2)*mean(intervals),'--k');
    limits = axis; axis([1,nvals,limits(3:4)])
end
%% figure 4 unsupervised learning
load('clval.mat')
load('ntDtm.mat')
figure(1)
subplot(221),
x = clpcavals;
c = [0 0 0; 0.6 0.6 0.6; 0.8 0.8 0.8];
f = {'square' '<' 'o'};
hold on
for i = 1:max(x(:,3))
    h2 = scatter(x(x(:,3)==i,1),x(x(:,3)==i,2));
    h2.Marker = f{i}; 
    h2.MarkerEdgeColor = [1 1 1];
    h2.MarkerFaceColor = c(i,:);
    h2.SizeData = 100;
end
xlabel('Coordinate 1');
ylabel('Coordinate 2');
matt = csvread('matthewTopics.csv',1,1)+rand(20,1)*.01;
subplot(223),h3 = bar(matt);
h3.EdgeColor = [1 1 1]; h3.FaceColor = [0 0 0];
xlim([0.1 20.9]); box off; xlabel('Topic'); ylabel('Posterior_{Matthew}')
%
hold on
% text(clpcavals(:,1)+.02,clpcavals(:,2),cldocs)
tree = linkage(dtm,'average','cosine');
figure(1)
subplot(122),
h1 = dendrogram(tree,'Labels',docs,'Orientation','left','ColorThreshold','default');
set(h1,'LineWidth',3);
xlabel('Height')
plotCorrect
col12(2)

% saveFig(strcat(wd,'\unsupervisedNt.tif'))
% saveFig(strcat(wd,'\unsupervisedNtGrey.tif'))
% saveFig(strcat(wd,'\unsupervisedNtGrey2.tif'))
%% clustegram for unsupervised learning k-means
d = pdist(dtm,'cosine');
d = squareform(d);
figure(2)
h2 = clustergram(d);
set(h2,'RowLabels',docs) 
set(h2,'ColumnLabels',docs) 
set(h2,'Colormap',bone)

a = findobj(gcf); % get the handles associated with the current figure
allaxes=findall(a,'Type','axes');
alllines = findall(a,'Type','line');
alltext = findall(a,'Type','text');
set(allaxes,'FontName','Times','FontWeight','Bold','LineWidth',1.5,...
'FontSize',12);
%% keywords
keywords = {'jesus' 'lord' 'god' 'man'};
%%retrieve ranking positions
for i = 1:length(keywords)
    ii = find(strcmp(ranked_vocabulary,keywords{i}));
    ranked_frequencies(ii)
end
% keyword distribution in one text
doc1 = fileread(filename{22});% matthew
% preprocess NT
temp1 = strtrim(regexprep(doc1,'([^\D]+)',' '));% remove numbers
temp1 = strtrim(regexprep(temp1,'([^\w]+)',' '));% non-alphabetic characters
temp1 = regexprep(['''',temp1,''''],' ',''','''); % insert '' and , around alphabetic sequence
evalc(['tokens1 = {',temp1,'}']); % assign tokens
doctokens1 = lower(tokens1);
% extract word frequencies
words1 = length(doctokens1);
[vocab1,void1,index1] = unique(doctokens1);
vocabwords1 = length(vocab1);
frequencies1 = hist(index1,vocabwords1);
[ranked_frequencies1,idx1] = sort(frequencies1,'descend');
ranked_vocabulary1 = vocab1(idx1);

plot(strcmp('jesus', doctokens1))
hold on
plot(strcmp('christ', doctokens1),'r')
%%
v = strcmp('jesus', doctokens1);
scatter(1:length(v),v,'.k')
hold on 
v = strcmp('god', doctokens1)*.9;
scatter(1:length(v),v,'.r')
%% count based-evaluation figure 3
keywords = {'jesus' 'christ' 'son' 'god' 'ghost' 'devil'};
V = zeros(length(keywords),length(doctokens1));
W = zeros(length(keywords),1);
c = bone(length(keywords) + 1); c = c(1:end-1,:);
hf = figure(1);
    set(hf,'PaperUnits','centimeters')
    xSize = 17; ySize = 12;
    xLeft = (21-xSize)/2; yTop = (30-ySize)/2;
    set(hf,'PaperPosition',[xLeft yTop xSize ySize])
    set(hf,'Position',[0 0 xSize*50 ySize*25])


%xSize = 17; ySize = 12;
set(hf,'Color',[1 1 1],'Name','Count-based evaluation')
%set(hf,'Position', [100, 100, 1049*.8, 895*.25])

% type-token
filename = getAllFiles(ddFull);
idx = regexp(filename{1},'\\');idx = idx(end)+1;
docname = cell(length(filename),1); for i = 1:length(filename); ...
        docname{i} = filename{i}(idx:end);end; 
docname = regexprep(docname,'\.txt','');
typ = sum(to ~= 0,2);
tok = sum(to,2);
TTR = (typ./tok)*100;
figure(1)
subplot(212)
h = bar(TTR);
h.FaceColor = [0 0 0];
ylabel('TTR%');
set(gca,'xtick',1:length(docname),'xticklabel',docname','xticklabelrotation',45)
plotCorrect
xlim([0.1 27.9])
box off
xlabel('Book')
% length issue
[r p] = corrcoef([typ tok TTR]);

% keyword distribution
subplot(221),
for i = 1:length(keywords)
    w = i/length(keywords);
    v = strcmp(keywords(i), doctokens1)*w;
    V(i,:) = v;
    W(i,1) = w;
    hold on
    h = scatter(1:length(v),v,'square','filled');
    % h.MarkerFaceColor = c(i,:);
    h.MarkerFaceColor = [0 0 0];
    h.MarkerEdgeColor = [0 0 0];
    h.SizeData = 20;
    % uppercase first letter for plot
    idx = regexp([' ' keywords{i}],'(?<=\s+)\S','start')-1;
    keywords{i}(idx) = upper(keywords{i}(idx));
    % add keyword
    h2 = text(0-5000,max(v)+.01,keywords(i));
    h2.FontName = 'Times'; h2.FontSize = 12; h2.FontWeight = 'bold';
    % h2.Color = c(i,:);
    h2.Color = [0 0 0];
end
set(gca,'ytick',W,'yticklabel',keywords)
set(gca,'xtick',[1 10000 20000],'xticklabel',{1 10000 20000})
ylim([.1 max(W)+.1])
xlim([0 length(v)+1])
xlabel('Word position')
ylabel('Keyword')
set(gca,'YColor','w')
% ## need cross correlation
    % V = double(V > 0);
    % [r p] = corrcoef(V')
    
% NER
cd(wd)
load('histRelativeMat.mat')
nerW = str2double(histRelativeMat(2:end,2:end));
%fh = figure(1);
subplot(222),
colormap(bone)
h1 = barh(nerW','stacked','linewidth',2);
hl = legend(histRelativeMat(2:end,1),'location','NorthEastOutside');
hl.FontSize = 8;
set(gca,'yticklabel',{'Pers' 'Org' 'Loc' 'Char'})
xlim([0 1.01])
ylim([0 5])
xlabel('Relative frequency')
legend boxoff
box off
col12(2)   
    
plotCorrect
%saveFig(strcat(wd,'/countEvaluation.tiff'))
%saveFig(strcat(wd,'/countEvaluationNolegend.tiff'))
%% ### relations between words
% build dtm
[dtm, vocabfull, docstokens, filename,~,~, dtmbi] = tdmVanilla(ddFull);
% ### co-occurrence matrix by transposed multiplication from binary dtm
dtmbi = sparse(double(dtmbi));
comat = dtmbi'*dtmbi;
% co-occurence of two words
i1 = strmatch({'jesus'},vocabfull,'exact'); 
i2 = strmatch({'said'},vocabfull,'exact');
probe = comat(i1,i2);
comat(i1,i1)    
% comat is sparse
nonsparse = sum(sum(comat > 0))/prod(size(comat));
% ### pointwise mutual information (PMI)
% individual word probabilities in corpus of books
p1wordbook = diag(comat)/length(filename);
% probability of encountering individual probes in books
disp(p1wordbook(i1))
    % book not containing 'jesus'
    idx = find(dtm(:,i1) == 0);
    disp(filename(idx))
disp(p1wordbook(i2))
% joint probabilities for words co-occurring in more than 10 books (n = 2)
[wa,wb,rawcount] = find(triu((comat > 25).*comat,1));
p2wordbook = rawcount/length(filename);
% compute pointwise mutual information
minfobook = log2(p2wordbook./(p1wordbook(wa).*p1wordbook(wb)));
% sort mutual information
[sortminfo,sortindex] = sort(minfobook,'descend');
% copy all wa and wb into char array according to I(wa,wb) order
tempa = char(vocabfull{wa(sortindex)});
tempb = char(vocabfull{wb(sortindex)});
% merge into single char array
tab = ones(size(tempa,1),1)*' '; 
temp = [tab,tempa,tab,tempb,tab];
% list the first n values for tempa, tempb and sortminfo
n = 100;
disp('-> Pairs of words with largest mutual information')
disp(' WORDa         WORDb          I(WORDa,WORDb)')
for k = 1:n;disp(sprintf('%s%6.4f',temp(k,:),sortminfo(k)));end
% list the last n values for tempa, tempb and sortminfo
n = 10;
disp('-> Pairs of words with lowest mutual information')
disp(' WORDa         WORDb          I(WORDa,WORDb)')
for k = 1:n;disp(sprintf('%s%6.4f',temp(end-k+1,:),sortminfo(end-k+1)));end
% sort information scores as they depart from zero
[void,absindex] = sort(abs(sortminfo));
% list the n words for tempa, tempb and sortminfo closest to zero
n = 10;
disp('-> Pairs of words with mutual information close to zero')
disp(' WORDa         WORDb          I(WORDa,WORDb)')
for k = 1:n
    disp(sprintf('%s%6.4f',temp(absindex(k),:),sortminfo(absindex(k))));
end
% calculaions for any two words
wStr = {'jesus' 'said'};% search words
rawOccur = diag(comat);
rawProb = diag(comat)/length(filename);
wO = zeros(size(wStr))';% word occurrence
wP = wO;% word probabilities 
inventory = vocabfull';
for i = 1:length(wStr)
    index(i) = find(strcmp(vocabfull,wStr{i}));
    wO(i,1) = rawOccur(index(i));
    wP(i,1) = rawProb(index(i)); 
end
wO(3,1) = comat(index(1),index(2)); % raw co-occurrence
wP(3,1) = comat(index(1),index(2))/length(filename); % joint probability 
wP(4,1) = wP(3)/wP(2);% wStr{1} conditioned on wStr{2}
wP(5,1) = wP(3)/wP(1);% wStr{2} conditioned on wStr{1}
wP(6,1) = log2(wP(3)/(wP(1)*wP(2)));% mutual information measured in bit
disp(wO)
disp(wP)
% ### geometric
% ##correlation between word vectors
i1 = strmatch({'jesus'},vocabfull,'exact'); 
i2 = strmatch({'said'},vocabfull,'exact');
w1 = dtm(:,i1);
w2 = dtm(:,i2);
% correlation distance between two words
[r p] = corrcoef(w1,w2);
% using a distance matrix
dmatcor = squareform(pdist(dtm','correlation'));
r = 1-dmatcor(i2,i1);  
% ## cosine distance
%One minus the cosine of the included angle between points (treated as vectors).
dmatcos = squareform(pdist(dtm','cosine'));
% cosine distance between two words
cosang = 1-dmatcos(i2,i1);  
%% numerical prediction
[dtm, vocabfull, docstokens, filename] = tdmVanilla(ddFull);
% ## repsonse variable
i1 = strmatch({'jesus'},vocabfull,'exact'); 
i2 = strmatch({'said'},vocabfull,'exact');
% document length
N = zeros(size(docstokens));
for i = 1:length(N); N(i) = length(docstokens{i});end;
% relative frequencies
j = dtm(:,i1)./N;
s = dtm(:,i2)./N;
% 'jesus + said' 
y = j+s;
% predictor
metaclass = {'paul','nonpaul','nonpaul','paul','paul','paul','nonpaul', ...
    'nonpaul','paul','paul','nonpaul','history','paul','paul','paul', ...
    'nonpaul','nonpaul','history','nonpaul','history','history', ...
    'history','paul','paul','nonpaul','paul','paul'};
x1 = categorical(metaclass);
% model
mdl1 = fitlm(x1,y);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
wd = 'C:\Users\KLN\Documents\textAnalysis\bible\newTestament';
ddFull = 'C:\Users\KLN\Documents\textAnalysis\bible\newTestament\ntTotal';
cd(wd);
%%
[to, vocabfull, docstokens, filename] = tdmVanilla(ddFull);
% Heaps' law --> needs work
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
