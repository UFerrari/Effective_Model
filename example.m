% Example code for inferring the Effective model introduced in 
% 'A simple model for low variability in neural spike trains', by Ferrari, Deny, Marre, Thierry, arXiv, 2018

clear all; close all;

%% LOADING DATA
% response = time-bins x repetitions matrix with the spike count
load data.mat

%% Inference

tolorance = 10^-6;
eta = 0.0;

Effective_Model = infer_Effective_Model( response_training , eta , tolorance);

%% Testing

mean_response = mean(response_testing,2);
var_response = var(response_testing,1,2);
lambdas = min( mean_response ) : 0.01 : max( mean_response );

figure
hold on
plot( mean_response , var_response , '.b', 'Markersize', 14) % Empirical Resppnse
plot( mean_response , mean_response , 'k','Linewidth',3.0); % Poisson model prediction
plot( lambdas, ppval( Effective_Model.var , lambdas ) , 'r','Linewidth',3.0); % Effective model prediction

