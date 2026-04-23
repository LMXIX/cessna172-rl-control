clear all; clc;

% 1. Initialize the Curriculum Environment
env = CessnaMasterEnvv2();
env.CurriculumPhase = 1;     % Phase 1: Agent = Elevator. PID = Aileron & Throttle.
env.TargetAltitude = 4000;
env.TargetSpeed = 100;

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% 2. Configure the PPO Agent
agentOpts = rlPPOAgentOptions(...
    'ExperienceHorizon', 3000, ...      % Learn from a full 60-second episode at once
    'ClipFactor', 0.1, ...              % REDUCED: Force smoother, more restrictive policy updates to prevent collapse
    'EntropyLossWeight', 0.005, ...     % REDUCED: Allow agent to settle into a smooth policy
    'MiniBatchSize', 512, ...           % INCREASED: Look at twice as much data before taking a step
    'NumEpoch', 3, ...
    'DiscountFactor', 0.995);           

% Auto-generate the Neural Network Brains
agent = rlPPOAgent(obsInfo, actInfo, agentOpts);

% Add gradient clipping to prevent network explosion
agent.AgentOptions.CriticOptimizerOptions.GradientThreshold = 1;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = 1;

agent.AgentOptions.CriticOptimizerOptions.LearnRate = 5e-5;
agent.AgentOptions.ActorOptimizerOptions.LearnRate = 5e-5; % Match critic so it memorizes the perfect strategy

% 3. Set up Training Parameters
% Re-adjusted mapping bounds because the Reward has been shaped to be much more dense
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 8000, ...            
    'MaxStepsPerEpisode', 3000, ...     
    'ScoreAveragingWindowLength', 50, ... % Average over the last 50 flights
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'EpisodeCount', ...
    'StopTrainingValue', 8000, ...       % Train for all 8000 episodes to show pure convergence
    'SaveAgentCriteria', 'EpisodeReward', ...
    'SaveAgentValue', 4500, ...          % Save any brilliant flight instantly
    'SaveAgentDirectory', 'curriculum_models/Phase1');  

% Create the save folder if it doesn't exist
if ~exist('curriculum_models/Phase1', 'dir')
    mkdir('curriculum_models/Phase1');
end

% 4. START THE SIMULATION
disp('Starting Phase 1 Curriculum: Pitch Control');
trainingStats = train(agent, env, trainOpts);

% Save the final converged champion explicitly
save('curriculum_models/Phase1/Champion_Phase1.mat', 'agent');

