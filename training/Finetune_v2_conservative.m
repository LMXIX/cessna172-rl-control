% =======================================================
% CONSERVATIVE FINE-TUNE: Agent2851 → Better altitude tracking
% while PRESERVING elevator smoothness.
%
% Changes from previous fine-tune:
%   - Altitude penalty: min(abs(err_h)/600, 0.1) — HALF strength
%   - Smoothness penalty: (delta^2)*10, capped at 3.0 — DOUBLED
%   - Stick movement: abs(delta)*1.0 — DOUBLED
%   - Ultra-tight clip (0.05) and very low entropy (0.0005)
%   - Lower learning rate (1e-5)
% =======================================================
clear all; clc; rng('shuffle');

%% Load Agent2851
disp('Loading Agent2851 (smooth baseline)...');
loaded = load('curriculum_models/Phase4/Agent2851.mat');
fields = fieldnames(loaded);
agent = loaded.(fields{1});
fprintf('Loaded Agent2851\n');

%% Setup environment (alpha=0.05, modified reward)
env = CessnaMasterEnvv2();
env.CurriculumPhase = 4;

%% Ultra-conservative hyperparameters
agent.AgentOptions.EntropyLossWeight = 0.0005;  % Minimal exploration
agent.AgentOptions.ClipFactor = 0.05;           % Tiny policy updates
agent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-5;
agent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-5;

%% Training
save_dir = 'curriculum_models/Phase4_finetune_v2';
if ~exist(save_dir, 'dir'), mkdir(save_dir); end

trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 3000, ...            
    'MaxStepsPerEpisode', 3000, ...     
    'ScoreAveragingWindowLength', 50, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'EpisodeCount', ...
    'StopTrainingValue', 3000, ...
    'SaveAgentCriteria', 'EpisodeReward', ...
    'SaveAgentValue', 2500, ...
    'SaveAgentDirectory', save_dir);  

disp('=== Conservative Fine-Tune: Gentle altitude fix + strong smoothness ===');
disp('  Altitude penalty: min(|err_h|/600, 0.1) — half strength');
disp('  Smoothness penalty: (delta^2)*10, cap 3.0 — doubled');
disp('  Stick movement: |delta|*1.0 — doubled');
trainingStats = train(agent, env, trainOpts);

save(fullfile(save_dir, 'Final_v2_Conservative.mat'), 'agent');
disp('Conservative fine-tune complete!');
