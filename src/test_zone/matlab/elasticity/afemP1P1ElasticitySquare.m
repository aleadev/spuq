function afemP1P1ElasticitySquare
% afemP1Poisson.m

    %% Initialization
    addpath(genpath(pwd));
    [c4n n4e n4sDb n4sNb] = loadGeometry('SquareNb3',5);
    minNrDoF = 1000;
    eta4nrDoF = sparse(1,1);
    
    %problem dependent parameters
    E = 3000;         %set by the user
    nu = 0.3;           %set by the user
    mu = E/(2*(1+nu)); 
    lambda = E*nu/((1+nu)*(1-2*nu));
    
    %% AFEM loop
    [x, nrDoF] = ...
        solveP1P1Elasticity(@f,@g,@u4Db,c4n,n4e,n4sDb,n4sNb,mu,lambda);
    %% Plot the solution
    plotP1P1(n4e,c4n,x,lambda,mu,20);
end

%% problem input data
function val = f(x)
    val = zeros(size(x,1),2);
end

function [W,M] = u4Db(x,lambda,mu)
    M = zeros(2*size(x,1),2);
    W = zeros(2*size(x,1),1);
    M(1:2:end,1) = 1;
    M(2:2:end,2) = 1;
%     value = u_value(x,lambda,mu);
%     W(1:2:end,1) = value(:,1);
%     W(2:2:end,1) = value(:,2);
    W(2:2:end) = x(:,1)*0.02;
end

function val = g(x,n)
    val = zeros(size(x,1),2);
end