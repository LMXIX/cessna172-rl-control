%% Curriculum Phase 2
% This script loads the Phase 1 Agent (which learned the basics of 
% wind rejection using 10% aileron authority) and expands its authority to 50%.

clear all; clc; rng('shuffle');

disp('Loading Pre-Trained Final_Phase2_Model Agent...');
% Dynamically load whatever variable is inside the mat file
loaded_data = load('curriculum_models/Phase2/Final_Phase2_Model.mat');
fields = fieldnames(loaded_data);
agent = loaded_data.(fields{1}); 

disp('Creating Simulation Environment (Phase 3)...');
env = CessnaMasterEnvv2();

% CRITICAL: Phase 3 Activates 50% Agent Authority / 50% PID
env.CurriculumPhase = 3;

% Obtain action and observation information
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% REFINING PPO HYPERPARAMETERS FOR EXPANDED AUTHORITY
% It already knows what to do, it just needs to recalibrate the 
% magnitude of its pushes. We maintain the ultra-slow learning rate
% so it doesn't violently overwrite its knowledge.
agent.AgentOptions.EntropyLossWeight = 0.001;  % Slight bump in exploration to test new limits
agent.AgentOptions.ClipFactor = 0.05;          % Allow slightly larger policy updates

agent.AgentOptions.CriticOptimizerOptions.LearnRate = 5e-5;
agent.AgentOptions.ActorOptimizerOptions.LearnRate = 5e-5; 

% TRAINING PARAMETERS
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 10000, ...            
    'MaxStepsPerEpisode', 3000, ...     
    'ScoreAveragingWindowLength', 50, ... 
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'EpisodeCount', ...
    'StopTrainingValue', 10000, ...      
    'SaveAgentCriteria', 'EpisodeReward', ...
    'SaveAgentValue', 4500, ...         
    'SaveAgentDirectory', 'curriculum_models/Phase3');  

% Create save folder if it doesn't exist
if ~exist('curriculum_models/Phase3', 'dir')
    mkdir('curriculum_models/Phase3');
end

trainingStats = train(agent, env, trainOpts);

disp('Training Complete. Saving Final Model...');
save('curriculum_models/Phase3/Final_Phase3_Model.mat', 'agent');
