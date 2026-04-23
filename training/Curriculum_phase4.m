%% Curriculum Phase 4: Full Robustness 
% This script loads the Phase 3 Agent 
% The agent now has 100% manual control over the Aileron and Throttle.

clear all; clc; rng('shuffle');

disp('Loading Pre-Trained Final_Phase3_Model Agent...');
loaded_data = load('curriculum_models/Phase3/Final_Phase3_Model.mat');
fields = fieldnames(loaded_data);
agent = loaded_data.(fields{1}); 

disp('Creating Simulation Environment (Phase 4)...');
env = CessnaMasterEnvv2();


env.CurriculumPhase = 4;

% Obtain action and observation information
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);


% The agent's Aileron and Elevator weights are perfect, but its Throttle 
% weights are complete random nonsense. We need to encourage enough exploration 
% for it to find the throttle, without overwriting the steering knowledge.
agent.AgentOptions.EntropyLossWeight = 0.005;  % HIGHER exploration for throttle
agent.AgentOptions.ClipFactor = 0.05;          

agent.AgentOptions.CriticOptimizerOptions.LearnRate = 5e-5;
agent.AgentOptions.ActorOptimizerOptions.LearnRate = 5e-5; 


trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 15000, ...            
    'MaxStepsPerEpisode', 3000, ...     
    'ScoreAveragingWindowLength', 50, ... 
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'EpisodeCount', ...
    'StopTrainingValue', 15000, ...      
    'SaveAgentCriteria', 'EpisodeReward', ...
    'SaveAgentValue', 4500, ...         
    'SaveAgentDirectory', 'curriculum_models/Phase4');  

% Create save folder if it doesn't exist
if ~exist('curriculum_models/Phase4', 'dir')
    mkdir('curriculum_models/Phase4');
end


disp('Starting Curriculum Phase 4 Training (100% Authority)...');
trainingStats = train(agent, env, trainOpts);

disp('Training Complete. Saving Final Model...');
save('curriculum_models/Phase4/Final_Phase4_Model.mat', 'agent');
