function [ Model ,hess] = infer_Effective_Model( response, eta,tolerance)
% [ Model ,hess] = infer_Effective_Model( PSTH, eta,threshold)
% Model = structure for the Effective neuron model
% hess = log-likelihood hessian at the solution
% response = T x R matrix integer spike counts (
% T=time-bin number, R=repetition number
% tolerance = numerical precision of the inference
% eta = L2 regularization on the inference parameters
%
% Use Newton method to infer the Effective model introduced in
% 'A simple model for low variability in neural spike trains', by Ferrari, Deny, Marre, Mora, arXiv, 2018
%
% The model characterize the sub-poissonian distribution of the spike-count sigma for a
% neuron with firing rate lambda:
% P_Effective(sigma | lambda ) exp( theta(lambda) * sigma + gamma * sigma^2
% + delta * sigma^3 - log( sigma! )  )/Z[lamda]
%
% [gamma,delta] = parameters of the model
% Z[lambda] = normalization
% theta(lambda) = linear input
%


% preparation and function declaration
sigmas = 0 : 2*max(response(:));
factSig = factorial(sigmas);
zed = @(th,gg,dd)  sum(  exp( th * sigmas + gg * sigmas.^2 + dd * sigmas.^3) ./ factSig );
logZed = @(th,gg,dd)  log( zed(th,gg,dd) );
meanOne   = @(th,gg,dd) sum( sigmas    .* exp( th * sigmas + gg * sigmas.^2  + dd * sigmas.^3 ) ./ factSig ) / zed(th,gg,dd);
meanTwo   = @(th,gg,dd) sum( sigmas.^2 .* exp( th * sigmas + gg * sigmas.^2  + dd * sigmas.^3 ) ./ factSig ) / zed(th,gg,dd) ;
meanThree = @(th,gg,dd) sum( sigmas.^3 .* exp( th * sigmas + gg * sigmas.^2  + dd * sigmas.^3 ) ./ factSig ) / zed(th,gg,dd) ;
meanFour  = @(th,gg,dd) sum( sigmas.^4 .* exp( th * sigmas + gg * sigmas.^2  + dd * sigmas.^3 ) ./ factSig ) / zed(th,gg,dd) ;
meanFive  = @(th,gg,dd) sum( sigmas.^5 .* exp( th * sigmas + gg * sigmas.^2  + dd * sigmas.^3 ) ./ factSig ) / zed(th,gg,dd) ;
meanSix   = @(th,gg,dd) sum( sigmas.^6 .* exp( th * sigmas + gg * sigmas.^2  + dd * sigmas.^3 ) ./ factSig ) / zed(th,gg,dd) ;
entropy   = @(th,gg,dd) -sum(  exp( th * sigmas + gg * sigmas.^2  + dd * sigmas.^3 ) ./ factSig  ./ zed(th,gg,dd) .* ( ( th * sigmas + gg * sigmas.^2  + dd * sigmas.^3 ) ./ factSig  - logZed(th,gg,dd) ) ); 

% observables estimation
oneList = mean(response   ,2);
twoList = mean(response.^2,2);
thrList = mean(response.^3,2);
twoList( oneList == 0 ) = [];
thrList( oneList == 0 ) = [];
oneList( oneList == 0 ) = [];

[oneUnique,~,index] = unique( oneList);

%regularization
reg = eta*eye(2);

% initial condition
gamma = -0.5;
delta = -0.1;
thetaUnique = arrayfun( @(ll) fzero( @(th) meanOne(th,gamma,delta)-ll  , log( ll ) ) , oneUnique);
theta = thetaUnique(index);

dataNd = sum( twoList );
dataRd = sum( thrList );
grad = [ dataNd - sum( arrayfun( @(th)  meanTwo(th,gamma,delta),theta ) ) ; dataRd - sum( arrayfun( @(th)  meanThree(th,gamma,delta),theta ) )] - reg * [gamma ; delta];

hess = zeros(2);
hess(1,1) = sum( arrayfun( @(th)  meanFour(th,gamma,delta) -meanTwo(th,gamma,delta)^2 ,theta ) );
hess(1,2) = sum( arrayfun( @(th)  meanFive(th,gamma,delta) -meanTwo(th,gamma,delta)*meanThree(th,gamma,delta) ,theta ) );
hess(2,1) = hess(1,2);
hess(2,2) = sum( arrayfun( @(th)  meanSix(th,gamma,delta) -meanThree(th,gamma,delta)^2 ,theta ) );


hess = hess + reg;


contraGrad = hess \ grad;
error = sqrt( 2*grad' * contraGrad/numel(oneList) );
    
alpha = 0.1;
error_Old = error;
count = 0;
while (error > tolerance) && (alpha>10^-4) 
    if mod(count,10)==0
        [mod(count,1000), log10(error), alpha, gamma,delta]
    end
    count = count+1;
    gamma = gamma + alpha * contraGrad(1);
    delta = delta + alpha * contraGrad(2);
    thetaUnique = arrayfun( @(ll) fzero( @(th) meanOne(th,gamma,delta)-ll  , log( ll ) ) , oneUnique);
    theta = thetaUnique(index);

    newGrad = [ dataNd - sum( arrayfun( @(th)  meanTwo(th,gamma,delta),theta ) ) ; dataRd - sum( arrayfun( @(th)  meanThree(th,gamma,delta),theta ) )] - reg * [gamma ; delta] ;
    hess(1,1) = sum( arrayfun( @(th)  meanFour(th,gamma,delta) -meanTwo(th,gamma,delta)^2 ,theta ) );
    hess(1,2) = sum( arrayfun( @(th)  meanFive(th,gamma,delta) -meanTwo(th,gamma,delta)*meanThree(th,gamma,delta) ,theta ) );
    hess(2,1) = hess(1,2);
    hess(2,2) = sum( arrayfun( @(th)  meanSix(th,gamma,delta) -meanThree(th,gamma,delta)^2 ,theta ) );
    hess = hess + reg;
    
    newContraGrad = hess \ newGrad;
    error = sqrt( 2 * newGrad' * newContraGrad /numel(oneList) );
    if error< error_Old
        alpha = alpha*1.04;
        grad = newGrad;
        contraGrad = newContraGrad;
        error_Old = error;
    else
        gamma = gamma - alpha * contraGrad(1);
        delta = delta - alpha * contraGrad(2);
        alpha = alpha/sqrt(1.3);
    end
end

% construction of the output structure

Model = struct();
lambdas = 10^-3:1/200:max(oneList);

%parameters
Model.params = [gamma,delta];

%linear input
theta = arrayfun( @(ll) fzero( @(th) meanOne(th,Model.params(1),Model.params(2))-ll  , log( ll / (1+ll ) ) ) , lambdas);
Model.theta = csape(  lambdas , [ 0 theta 0] , [2 2]);

%variance
varPred_Gamma = arrayfun( @(ll) meanTwo( ppval( Model.theta , ll) , Model.params(1),Model.params(2)) - ll^2 , lambdas);
Model.var = csape(  lambdas , [ 0 varPred_Gamma 0] , [2 2]);

%log normalization
zed =  arrayfun( @(ll) logZed(ppval( Model.theta , ll), Model.params(1), Model.params(2) )  , lambdas);
Model.logZed = csape(  lambdas , [ 0 zed 0] , [2 2]);

%entropy
Model.S = csape(  lambdas , [ 0 arrayfun(@(ll) entropy( ppval( Model.theta , ll) , Model.params(1) , Model.params(2) )   , lambdas) 0 ] , [2 2]);
end

