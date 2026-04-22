%% Curriculum Phase 2: Wind Disturbances
% This script loads the perfectly converged Phase 1 Still-Air Agent
% and begins training it against severe environmental wind turbulence.

clear all; clc; rng('shuffle');

disp('Loading Pre-Trained Champion_Phase1 Agent...');
% Dynamically load whatever variable is inside the mat file
loaded_data = load('curriculum_models/Phase1/Champion_Phase1.mat');
fields = fieldnames(loaded_data);
agent = loaded_data.(fields{1}); 

disp('Creating Simulation Environment (Phase 2)...');
env = CessnaMasterEnvv2();

% CRITICAL: Phase 2 Activates Domain Randomization & Wind
env.CurriculumPhase = 2;

% Obtain action and observation information
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% -------------------------------------------------------------
% REFINING PPO HYPERPARAMETERS FOR EXISTING KNOWLEDGE
% -------------------------------------------------------------
% The agent is already an expert at flying. We DO NOT want it to 
% "explore" by crashing the plane. We want it to politely adapt its 
% existing smooth control laws to fight the wind.
agent.AgentOptions.EntropyLossWeight = 0.0005; % Barely any random exploration
agent.AgentOptions.ClipFactor = 0.02;          % Do not allow massive policy updates

% ULTRA SLOW LEARNING RATE: 1e-5. 
% We want to carefully curve the existing weights, not overwrite them.
agent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-5;
agent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-5; 

% -------------------------------------------------------------
% TRAINING PARAMETERS
% -------------------------------------------------------------
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 20000, ...            
    'MaxStepsPerEpisode', 3000, ...     
    'ScoreAveragingWindowLength', 50, ... 
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'EpisodeCount', ...
    'StopTrainingValue', 20000, ...      % Train for all 8000 episodes
    'SaveAgentCriteria', 'EpisodeReward', ...
    'SaveAgentValue', 4500, ...         % Save any agent that masters the wind
    'SaveAgentDirectory', 'curriculum_models/Phase2');  

% Create save folder if it doesn't exist
if ~exist('curriculum_models/Phase2', 'dir')
    mkdir('curriculum_models/Phase2');
end

% -------------------------------------------------------------
% START TRAINING
% -------------------------------------------------------------
disp('Starting Curriculum Phase 2 Training (Wind Turbulence active)...');
trainingStats = train(agent, env, trainOpts);

disp('Training Complete. Saving Final Model...');
save('curriculum_models/Phase2/Final_Phase2_Model.mat', 'agent');
