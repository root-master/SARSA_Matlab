%function [weights,data,convergence] = sarsa_gridworld(task,functionApproximator,withBias,s_end,weights,data,nMeshx,nTilex,nMeshy,nTiley)
clc, close all, clear all;
withBias = false;

nMeshx = 20; nMeshy = 20;
nTilex = 1; nTiley = 1;

functionApproximator = 'kwtaNN';
% control task could be 'grid_world' or 'puddle_world'
task = 'puddle_world';
% function approximator can be either 'kwtaNN' or 'regularBPNN'


% goal in continouos state
s_end = [1.0,1.0];

% Input of function approximator
xgridInput = 1.0 / nMeshx;
ygridInput = 1.0 / nMeshy;
xInputInterval = 0 : xgridInput : 1.0;
yInputInterval = 0 : ygridInput : 1.0;

% the number of states -- This is the gross mesh states ; the 1st tiling 
nStates = length(xInputInterval) * length(yInputInterval); 

% on each grid we can choose from among this many actions 
% [ up , down, right, left ]
% (except on edges where this action is reduced): 
nActions = 4; 

%% kwta and regular BP Neural Network
% Weights from input (x,y) to hidden layer
InputSize = length(xInputInterval) + length(yInputInterval);
nCellHidden = round(0.5 * nStates);
mu = 0.1;
Wih = mu * (rand(InputSize,nCellHidden) - 0.5);
biasih = mu * ( rand(1,nCellHidden) - 0.5 );
% biasih = zeros(1,nCellHidden);
% Weights from hidden layer to output
Who = mu * (rand(nCellHidden,nActions) - 0.5);
biasho = mu * ( rand(1,nActions) - 0.5 );
% biasho = zeros(1,nActions);

%% Linear Neural Net
mu = 0.1; % amplitude of random weights
Wio = mu * (rand(InputSize,nActions) - 0.5);
biasio = mu * (rand(1,nActions) - 0.5 );

%% RUN
if withBias,
    wB = 'withBias';
else
    wB = 'withoutBias';
end

alpha = 0.005;

% on each grid we can choose from among this many actions 
% [ up , down, right, left ]
% (except on edges where this action is reduced): 
nActions = 4; 

gamma = 0.99;    % discounted task 
epsilon = 0.1;  % epsilon greedy parameter


% Max number of iteration in ach episde to break the loop if AGENT
% can't reach the GOAL 
maxIteratonEpisode = 2 * (nMeshx * nTilex + nMeshy * nTiley);


shunt = 1;             


%%
% Input of function approximator
xgridInput = 1.0 / nMeshx;
ygridInput = 1.0 / nMeshy;
xInputInterval = 0 : xgridInput : 1.0;
yInputInterval = 0 : ygridInput : 1.0;
% smoother state space with tiling
xgrid = 1 / (nMeshx * nTilex);
ygrid = 1 / (nMeshy * nTiley);
% parameter of Gaussian Distribution
sigmax = 1.0 / nMeshx; 
sigmay = 1.0 / nMeshy;

xVector = 0:xgrid:1;
yVector = 0:ygrid:1;

nStates = length(xInputInterval) * length(yInputInterval);

%% Different Max number of episodes
maxNumEpisodes = 100 * nStates * nTilex * nTiley;

nGoodEpisodes = 0; % a variable for checking the convergence
convergence = false;
agentReached2Goal = false;
agentBumped2wall = false;
%% Episode Loops
ei = 1;
deltaForStepsOfEpisode = [];

while (ei < maxNumEpisodes && ~convergence ), % ei<maxNumEpisodes && % ei is counter for episodes
     % initialize the starting state - Continuous state
     s = initializeState(xVector,yVector);
     x = s(1); y = s(2);
     % Gaussian Distribution on continuous state
     xt = sigmax * sqrt(2*pi) * normpdf(xInputInterval,x,sigmax);
     yt = sigmay * sqrt(2*pi) * normpdf(yInputInterval,y,sigmay);
     % Using st as distributed input for function approximator
     st = [xt,yt];     
          
     % initializing time
     ts = 1;
     [Q,h,id] = kwta_NN_forward_new(st,shunt,Wih,biasih,Who,biasho);
      act = e_greedy_policy(Q,nActions,epsilon);

    %% Episode While Loop
    while( ~(s(1)==s_end(1) && s(2)==s_end(2)) && ts < maxIteratonEpisode),
        % update state to state+1
        sp1 = UPDATE_STATE(s,act,xgrid,xVector,ygrid,yVector);
        xp1 = sp1(1); yp1 = sp1(2);
        % PDF of stp1
        xtp1 = sigmax * sqrt(2*pi) * normpdf(xInputInterval,xp1,sigmax);
        ytp1 = sigmay * sqrt(2*pi) * normpdf(yInputInterval,yp1,sigmay);
        stp1=[xtp1,ytp1];
        
        if ( sp1(1)==s_end(1) && sp1(2)==s_end(2) ),
            agentReached2Goal = true;
            agentBumped2wall = false;
        elseif ( sp1(1)==s(1) && sp1(2)==s(2) ),
            agentBumped2wall = true;
            agentReached2Goal = false;
        else
            agentBumped2wall = false;
            agentReached2Goal = false;            
        end
        
        % reward/punishment from Environment
        rew = ENV_REWARD(sp1,agentReached2Goal,agentBumped2wall,nTilex,nTiley);
        [Qp1,hp1,idp1] = kwta_NN_forward_new(stp1,shunt,Wih,biasih,Who,biasho);
        
        % make the greedy action selection in st+1: 
        actp1 = e_greedy_policy(Qp1,nActions,epsilon);
    
        if( ~agentReached2Goal ) 
            % stp1 is not the terminal state
            delta = rew + gamma * Qp1(actp1) - Q(act);
            deltaForStepsOfEpisode = [deltaForStepsOfEpisode,delta];           
            % Update Neural Net
            [Wih,biasih,Who,biasho] = Update_kwtaNN_new(st,act,h,id,alpha,delta,Wih,biasih,Who,biasho,withBias);           
        else
            % stp1 is the terminal state ... no Q(s';a') term in the sarsa update
            fprintf('Reaching to Goal at episode =%d at step = %d and mean(delta) = %f \n',ei,ts,mean(deltaForStepsOfEpisode));
            delta = rew - Q(act);
            [Wih,biasih,Who,biasho] = Update_kwtaNN_new(st,act,h,id,alpha,delta,Wih,biasih,Who,biasho,withBias);
            break; 
        end
        % update (st,at) pair:
        st = stp1;  s = sp1; act = actp1;
        Q = Qp1;    
        ts = ts + 1;
    end % while loop
    meanDeltaForEpisode(ei) = mean(deltaForStepsOfEpisode);
    varianceDeltaForEpisode(ei) =var(deltaForStepsOfEpisode);
    stdDeltaForEpisode(ei) = std(deltaForStepsOfEpisode);
    
    
    if ( ei>500 && abs(meanDeltaForEpisode(ei))< 0.2 && agentReached2Goal ),
            %&& abs(meanDeltaForEpisode(ei))<abs(meanDeltaForEpisode(ei-1) ) ),
        epsilon = bound(epsilon * 0.999,[0.001,0.1]);
    else
        epsilon = bound(epsilon * 1.01,[0.001,0.1]);
    end
    
    if ( abs(meanDeltaForEpisode(ei))<0.1 ) && agentReached2Goal,
        nGoodEpisodes = nGoodEpisodes + 1;
    else
        nGoodEpisodes = 0;
    end
    
    if  abs(mean(deltaForStepsOfEpisode))<0.05 && nGoodEpisodes> nStates*nTilex*nTiley,
        convergence = true;
        fprintf('Convergence at episode: %d \n',ei);
    end
    
    
%     plot(meanDeltaForEpisode)      
%     title(['Episode: ',int2str(ei),' epsilon: ',num2str(epsilon)])    
%     drawnow    
    ei = ei + 1;

end  % end episode loop

%% Saving weights and performance data in an structure
weights = struct;
data = struct;
data.meanDeltaForEpisode = meanDeltaForEpisode;
data.varianceDeltaForEpisode = varianceDeltaForEpisode;
data.stdDeltaForEpisode = stdDeltaForEpisode;

weights.Wih = Wih;
weights.biasih = biasih;
weights.Who = Who;
weights.biasho = biasho;
weights.biasio = biasio;
weights.Wio = Wio;

%% Plot Q values and Policy Map
grafica = 1;
[Qestimate] = plotQ_PolicyMap(s_end,nMeshx,xVector,xInputInterval,nMeshy,yVector,yInputInterval,weights,functionApproximator,grafica);

