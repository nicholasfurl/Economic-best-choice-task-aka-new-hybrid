function run_io;

%1 dummy 2 pid	3 reward_cond	4 num_options 5 PriorMean	6 PriorVar	7 option01	option02	option03	option04	option05	option06	option07	option08	option09	option10	option11	option12
data = table2array(readtable('C:\matlab_files\jspych\fiance_hybrid\pilot_analysis\sequence_subjVals.csv'));

subs = unique(data(:,2));
for subject = 1:numel(subs)
    
    this_data = data(data(:,2)==subs(subject),:);
    
    for sequence = 1:size(this_data,1);
        
        num_options = this_data(sequence,4);
        reward_cond = this_data(sequence,3);
        
        prior.mu = this_data(sequence,5);
        prior.sig = this_data(sequence,6);
        prior.kappa= 2;
        prior.nu = 1;
        
        
        list.vals = this_data(sequence,7:7+num_options-1);
        list.length = num_options;
        list.Cs = 0;
        list.payoff_scheme = -1;
        
        disp(list.vals)
        
        [choiceCont, choiceStop, difVal] = analyzeSecretaryNick_python(prior,list);
        
        num_samples(sequence,subject) = find(difVal<0,1,'first');
        
        
    end;    %sequences
    
    
end;    %subs

disp('audi5000')


fprintf(' ');



function [choiceCont, choiceStop, difVal] = analyzeSecretaryNick_python(priorProb,list)

sampleSeries = list.vals;
N = list.length;
Cs = list.Cs;
payoff_scheme = list.payoff_scheme;

sdevs = 8;
dx = 2*sdevs*sqrt(priorProb.sig)/100;
x = ((priorProb.mu - sdevs*sqrt(priorProb.sig)) + dx : dx : ...
    (priorProb.mu + sdevs*sqrt(priorProb.sig)))';


Nconsider = N;

difVal = zeros(1, Nconsider);
choiceCont = zeros(1, Nconsider);
choiceStop = zeros(1, Nconsider);

for ts = 1 : Nconsider
    
    [expectedStop, expectedCont] = rnkBackWardInduction(sampleSeries, ts, priorProb, N, x, Cs, payoff_scheme);
    
    if ts == 12;
  %      fprintf('');
    end;
    
    difVal(ts) = expectedCont(ts) - expectedStop(ts);
    
    choiceCont(ts) = expectedCont(ts);
    choiceStop(ts) = expectedStop(ts);
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [expectedStop, expectedCont, expectedUtility] = rnkBackWardInduction(sampleSeries, ts, priorProb, ...
    listLength, x, Cs, payoff_scheme)


N = listLength;
Nx = length(x);



if payoff_scheme == 1;  %if instruction are to maximise option value
    
    payoff = sort(sampleSeries,'descend'); %maximise all option values
    % payoff = [N:-1:1];    %maximize all option ranks
    
else;   %if payoff scheme is to get some degree of reward for each of top three ranks
    
    payoff = [.12 .08 .04];
    
end;



maxPayRank = numel(payoff);
payoff = [payoff zeros(1, 20)];

data.n  = ts;

data.sig = var(sampleSeries(1:ts));
data.mu = mean(sampleSeries(1:ts));

utCont  = zeros(length(x), 1);
utility = zeros(length(x), N);

if ts == 0
    ts = 1;
end

[rnkvl, rnki] = sort(sampleSeries(1:ts), 'descend');
z = find(rnki == ts);
rnki = z;

ties = 0;
if length(unique(sampleSeries(1:ts))) < ts
    ties = 1;
end

mxv = ts;
if mxv > maxPayRank
    mxv = maxPayRank;
end

rnkv = [Inf*ones(1,1); rnkvl(1:mxv)'; -Inf*ones(20, 1)];
% rnkv = [Inf*ones(1,1); rnkvl(1:mxv); -Inf*ones(20, 1)];

if ts == 4
%    fprintf('');
end;

[postProb] = normInvChi(priorProb, data);
px = posteriorPredictive(x, postProb);
px = px/sum(px);

Fpx = cumsum(px);
cFpx = 1 - Fpx;

for ti = N : -1 : ts
    
   % disp(sprintf('%d %d',ts, ti))
    
    if ti == N
        utCont = -Inf*ones(Nx, 1);
    elseif ti == ts
        utCont = ones(Nx, 1)*sum(px.*utility(:, ti+1));
    else
        utCont = computeContinue(utility(:, ti+1), postProb, x, ti);
    end
    
    %%%% utility when rewarded for best 3, $5, $2, $1
    utStop = NaN*ones(Nx, 1);
    
    rd = N - ti; %%% remaining draws
    id = max([(ti - ts - 1) 0]); %%% intervening draws
    td = rd + id;
    ps = zeros(Nx, maxPayRank);
    
    for rk = 0 : maxPayRank-1
        
        pf = prod(td:-1:(td-(rk-1)))/factorial(rk);
        
        ps(:, rk+1) = pf*(Fpx.^(td-rk)).*((cFpx).^rk);
        
    end
    
    for ri = 1 : maxPayRank+1
        
        z = find(x < rnkv(ri) & x >= rnkv(ri+1));
        utStop(z) = ps(z, 1:maxPayRank)*(payoff(1+(ri-1):maxPayRank+(ri-1))');
        
    end
    
    if ts == 4 & ti == 12
 %       fprintf('');
    end;
    
    if sum(isnan(utStop)) > 0
 %       fprintf('Nan in utStop');
    end
    
    if ti == ts
        [zv, zi] = min(abs(x - sampleSeries(ts)));
        if zi + 1 > length(utStop)
            zi = length(utStop) - 1;
        end
        
        utStop = utStop(zi+1)*ones(Nx, 1);
        
    end
    
    utCont = utCont - Cs;
    
    utility(:, ti)      = max([utStop utCont], [], 2);
    expectedUtility(ti) = px'*utility(:,ti);
    
    expectedStop(ti)    = px'*utStop;
    expectedCont(ti)    = px'*utCont;
    
    if ts == 12 & ti == 12
 %       fprintf('');
    end;
    
end

%fprintf('')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function utCont = computeContinue(utility, postProb0, x, ti)

postProb0.nu = ti-1;

utCont = zeros(length(x), 1);

% pspx = zeros(length(x), length(x));

expData.n   = 1;
expData.sig = 0;

for xi = 1 : length(x)
    
    expData.mu  = x(xi);
    
    postProb = normInvChi(postProb0, expData);
    spx = posteriorPredictive(x, postProb);
    spx = (spx/sum(spx));
    
    %     pspx(:, xi) = spx;
    
    utCont(xi) = spx'*utility;
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [postProb] = normInvChi(prior, data)

postProb.nu    = prior.nu + data.n;

postProb.kappa = prior.kappa + data.n;

postProb.mu    = (prior.kappa/postProb.kappa)*prior.mu + (data.n/postProb.kappa)*data.mu;

postProb.sig   = (prior.nu*prior.sig + (data.n-1)*data.sig + ...
    ((prior.kappa*data.n)/(postProb.kappa))*(data.mu - prior.mu).^2)/postProb.nu;

if data.n == 0
    postProb.sig = prior.sig;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function prob_y = posteriorPredictive(y, postProb)

tvar = (1 + postProb.kappa)*postProb.sig/postProb.kappa;

sy = (y - postProb.mu)./sqrt(tvar);

prob_y = tpdf(sy, postProb.nu);

