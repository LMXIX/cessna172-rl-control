% FYP Scenario 1: 500ft Step-Climb Transient Response
% Strict Comparison: Simulink PID vs RL Agent (Phase 4)

clear all; clc; close all;

t_end = 120;      
dt = 0.02;          
n_steps = floor(t_end / dt);
t_vec = linspace(0, t_end, n_steps);

% Create the 500ft step command (at 10 seconds)
cmd_h = 4000 * ones(1, n_steps);
cmd_h(t_vec >= 10) = 4500; 

% Load the Trained RL Agent
disp('Loading Phase 4 Agent...');
if exist('curriculum_models/Phase4_finetune/Final_v2_Finetuned.mat', 'file')
    load('curriculum_models/Phase4_finetune/Final_v2_Finetuned.mat', 'agent');
else
    files = dir('curriculum_models/Phase4/Agent*.mat');
    if isempty(files)
        error('No saved agents found in Phase4 folder.');
    end
    [~, idx] = max([files.datenum]);
    latest_agent = fullfile(files(idx).folder, files(idx).name);
    load(latest_agent, 'agent');
end
actor = getActor(agent);

% init jsbsim phase 4 env
env = CessnaMasterEnvv2();
env.CurriculumPhase = 4;
fdm = env.fdm;


% run the simulink baseline
simOut = sim('Cessna_PID_v1', 'StopTime', num2str(t_end));

pid_time = simOut.tout;
pid_alt = squeeze(simOut.sim_alt);
pid_v = squeeze(simOut.sim_v) * 0.592484; % convert to kts
pid_theta = rad2deg(squeeze(simOut.sim_theta));
pid_elev = squeeze(simOut.sim_elev);


%% RL agent
env.TargetAltitude = 4000;
env.TargetSpeed = 100;

% Force spawn for fair comparison
obs = env.reset(); 
fdm.set_property_value('atmosphere/turbulence/milspec/wind-severity', 0); 
fdm.set_property_value('ic/h-sl-ft', 4000);
fdm.set_property_value('ic/vc-kts', 100);
fdm.run_ic();

% Spool up engine identically
fdm.set_property_value('fcs/throttle-cmd-norm', 0.80);
for settle_idx = 1:25
    fdm.run();
end

data_rl = zeros(n_steps, 5);

for i = 1:n_steps
    target_alt = cmd_h(i);
    env.TargetAltitude = target_alt;
    
    % Step 1: Re-calculate the observation because the target changes mid-flight
    h = double(fdm.get_property_value('position/h-sl-ft'));
    h_dot = double(fdm.get_property_value('velocities/h-dot-fps'));
    theta = double(fdm.get_property_value('attitude/theta-rad'));
    q = double(fdm.get_property_value('velocities/q-rad_sec'));
    phi = double(fdm.get_property_value('attitude/phi-rad'));
    p = double(fdm.get_property_value('velocities/p-rad_sec'));
    v_kts = double(fdm.get_property_value('velocities/vc-kts'));
    elev_pos = env.fdm.get_property_value('fcs/elevator-cmd-norm');
    
    err_h = h - target_alt;
    err_v = v_kts - env.TargetSpeed;
    
    % Assemble correct observation vector for Env v2:
    % [err_h, h_dot, theta, q, phi, p, err_v, elev_pos]
    obs = [err_h/1000; h_dot/100; theta; q; phi; p; err_v/10; elev_pos];
    
    % Step 2: Agent decides action
    action = evaluate(actor, {obs});
    act_num = cell2mat(action);
    
    % Step 3: Advance environment
    [obs, ~, ~, ~] = env.step(act_num);
    
    % Step 4: Log physics outputs
    theta_out = double(fdm.get_property_value('attitude/theta-rad'));
    elev_out = env.fdm.get_property_value('fcs/elevator-cmd-norm');
    
    % [h, v, pitch, elev_act, ail_act]
    data_rl(i, :) = [h, v_kts, rad2deg(theta_out), elev_out, act_num(2)]; 
end


% generate the plots for the report
% colors to match my other graphs
color_pid = [0.33, 0.49, 0.74]; 
color_rl  = [0.42, 0.71, 0.43]; 

figure('Position', [100, 100, 950, 700]);
set(gcf, 'color', 'w');

subplot(2,2,1);
plot(pid_time, pid_alt, 'Color', color_pid, 'LineWidth', 1.6); hold on;
plot(t_vec, data_rl(:,1), 'Color', color_rl, 'LineWidth', 1.6);
plot(t_vec, cmd_h, 'k--', 'LineWidth', 1.2); 
ylabel('Altitude (ft)'); title('Altitude Response');
legend('PID', 'PPO Agent', 'Target (4500 ft)', 'Location', 'best', 'FontSize', 10); 
grid on; xlim([0 t_end]); xlabel('Time (s)');

subplot(2,2,2);
plot(pid_time, pid_v, 'Color', color_pid, 'LineWidth', 1.6); hold on;
plot(t_vec, data_rl(:,2), 'Color', color_rl, 'LineWidth', 1.6);
ylabel('Airspeed (kts)'); title('Airspeed Response'); 
grid on; xlim([0 t_end]); xlabel('Time (s)');

subplot(2,2,3);
plot(pid_time, pid_theta, 'Color', color_pid, 'LineWidth', 1.4); hold on;
plot(t_vec, data_rl(:,3), 'Color', color_rl, 'LineWidth', 1.4);
ylabel('Pitch Angle (\circ)'); title('Pitch Attitude'); 
grid on; xlim([0 t_end]); xlabel('Time (s)');

subplot(2,2,4);
plot(pid_time, pid_elev, 'Color', color_pid, 'LineWidth', 1.4); hold on;
plot(t_vec, data_rl(:,4), 'Color', color_rl, 'LineWidth', 1.4); 
ylabel('Elevator (norm)'); ylim([-1 1]); 
title('Elevator Actuation'); 
grid on; xlim([0 t_end]); xlabel('Time (s)');

sgtitle('500 ft Step-Climb Transient Response', 'FontSize', 12, 'FontWeight', 'bold');

saveas(gcf, 'physicstest_comparison.png');