% severity_sweep_monte_carlo.m
% Monte Carlo sweep of turbulence severities.
% Runs multiple random seeds for each severity to ensure statistical validity.
clear all; clc; close all;

t_end = 120; 
dt = 0.02;
n_steps = floor(t_end / dt);
t_vec = linspace(0, t_end, n_steps);

severities = [1, 2, 3, 4, 5];
n_sev = length(severities);
N_RUNS = 10; % 10 monte carlo seeds per severity (gives robust statistical margin)

rms_alt_pid = zeros(n_sev, N_RUNS);
rms_alt_rl  = zeros(n_sev, N_RUNS);
max_alt_pid = zeros(n_sev, N_RUNS);
max_alt_rl  = zeros(n_sev, N_RUNS);

base_seed = 1000;

% load agent
disp('Loading Fine-Tuned v2 Agent...');
loaded = load('curriculum_models/Phase4_finetune/Final_v2_Finetuned.mat');
fields = fieldnames(loaded);
agent = loaded.(fields{1});
actor = getActor(agent);

% PID Gains
kp_h = 0.008; ki_h = 0.002; kd_h = 0.018;  
kp_theta = 2.5; kd_q = 0.8;     
kp_psi = 0.5; kp_phi = 1.5; ki_phi = 0.1; kd_p = 0.2;
POLARITY = -1.0;

for s = 1:n_sev
    sev = severities(s);
    disp(['=== SEVERITY ' num2str(sev) ' ===']);
    for r = 1:N_RUNS
        seed = base_seed + r;
        
        % --- PID RUN ---
        env_pid = CessnaMasterEnv();
        fdm_pid = env_pid.fdm;
        env_pid.TargetAltitude = 4000;
        env_pid.reset();
        
        fdm_pid.set_property_value('ic/h-sl-ft', 4000);
        fdm_pid.set_property_value('ic/vc-kts', 100);
        fdm_pid.run_ic();
        
        fdm_pid.set_property_value('propulsion/tank[0]/contents-lbs', 150);
        fdm_pid.set_property_value('propulsion/tank[1]/contents-lbs', 150);
        fdm_pid.set_property_value('fcs/mixture-cmd-norm', 0.85);
        fdm_pid.set_property_value('propulsion/magneto_cmd', 3);
        fdm_pid.set_property_value('propulsion/starter_cmd', 1);
        fdm_pid.set_property_value('propulsion/engine[0]/set-running', 1);
        
        fdm_pid.set_property_value('atmosphere/turb-type', 4);
        fdm_pid.set_property_value('atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps', 20);
        fdm_pid.set_property_value('atmosphere/turbulence/milspec/severity', sev);
        fdm_pid.set_property_value('atmosphere/turbulence/milspec/seed', seed);
        fdm_pid.set_property_value('fcs/throttle-cmd-norm', 0.80);
        for k=1:25, fdm_pid.run(); end
        
        alt_pid = zeros(n_steps, 1);
        int_h = 0; int_phi = 0;
        
        for i=1:n_steps
            h = double(fdm_pid.get_property_value('position/h-sl-ft'));
            h_dot = double(fdm_pid.get_property_value('velocities/h-dot-fps'));
            v = double(fdm_pid.get_property_value('velocities/vc-kts'));
            theta = double(fdm_pid.get_property_value('attitude/theta-rad'));
            phi = double(fdm_pid.get_property_value('attitude/phi-rad'));
            q = double(fdm_pid.get_property_value('velocities/q-rad_sec'));
            p = double(fdm_pid.get_property_value('velocities/p-rad_sec'));
            psi = double(fdm_pid.get_property_value('attitude/psi-rad'));
            if psi > pi, psi = psi - 2*pi; end
            
            err_v = 100 - v;
            thr_cmd = max(min(0.65 + (0.02 * err_v), 1.0), 0.0);
            fdm_pid.set_property_value('fcs/throttle-cmd-norm', thr_cmd);
            
            err_h = 4000 - h;
            int_h = max(min(int_h + (err_h * dt), 20), -20);
            theta_cmd = max(min((kp_h*err_h) + (ki_h*int_h) - (kd_h*h_dot), 0.087), -0.15);
            elev = max(min(POLARITY * (kp_theta*(theta_cmd-theta) - kd_q*q), 1), -1);
            
            err_psi = 0 - psi;
            phi_cmd = max(min(kp_psi * err_psi, 0.35), -0.35);
            err_phi = phi_cmd - phi;
            int_phi = max(min(int_phi + (err_phi * dt), 0.5), -0.5);
            ail = max(min((kp_phi*err_phi) + (ki_phi*int_phi) - (kd_p*p), 1), -1);
            
            fdm_pid.set_property_value('fcs/elevator-cmd-norm', elev);
            fdm_pid.set_property_value('fcs/aileron-cmd-norm', ail);
            fdm_pid.run();
            alt_pid(i) = h;
        end
        % skip first 5s for transients
        settle = floor(5/dt);
        err_pid = alt_pid(settle:end) - 4000;
        rms_alt_pid(s, r) = sqrt(mean(err_pid.^2));
        max_alt_pid(s, r) = max(abs(err_pid));
        
        % --- RL AGENT RUN ---
        env = CessnaMasterEnvv2();
        env.CurriculumPhase = 4;
        fdm = env.fdm;
        env.TargetAltitude = 4000;
        env.TargetSpeed = 100;
        env.reset();
        
        fdm.set_property_value('ic/h-sl-ft', 4000);
        fdm.set_property_value('ic/vc-kts', 100);
        fdm.run_ic();
        
        fdm.set_property_value('propulsion/tank[0]/contents-lbs', 150);
        fdm.set_property_value('propulsion/tank[1]/contents-lbs', 150);
        fdm.set_property_value('fcs/mixture-cmd-norm', 0.85);
        fdm.set_property_value('propulsion/magneto_cmd', 3);
        fdm.set_property_value('propulsion/starter_cmd', 1);
        fdm.set_property_value('propulsion/engine[0]/set-running', 1);
        
        fdm.set_property_value('atmosphere/turb-type', 4);
        fdm.set_property_value('atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps', 20);
        fdm.set_property_value('atmosphere/turbulence/milspec/severity', sev);
        fdm.set_property_value('atmosphere/turbulence/milspec/seed', seed);
        for k=1:25, fdm.run(); end
        
        alt_rl = zeros(n_steps, 1);
        for i=1:n_steps
            h_current = double(fdm.get_property_value('position/h-sl-ft'));
            h_dot_current = double(fdm.get_property_value('velocities/h-dot-fps'));
            theta_current = double(fdm.get_property_value('attitude/theta-rad'));
            q_current = double(fdm.get_property_value('velocities/q-rad_sec'));
            phi_current = double(fdm.get_property_value('attitude/phi-rad'));
            p_current = double(fdm.get_property_value('velocities/p-rad_sec'));
            v_kts_current = double(fdm.get_property_value('velocities/vc-kts'));
            elev_pos_current = double(fdm.get_property_value('fcs/elevator-cmd-norm'));
            
            err_h_c = h_current - env.TargetAltitude;
            err_v_c = v_kts_current - env.TargetSpeed;
            obs = [err_h_c/1000; h_dot_current/100; theta_current; q_current; phi_current; p_current; err_v_c/10; elev_pos_current];
            
            action = evaluate(actor, {obs});
            act_num = cell2mat(action);
            
            elev = max(min(act_num(1), 1), -1);
            ail = max(min(act_num(2), 1), -1);
            thr_raw = max(min(act_num(3), 1), -1);
            thr = (thr_raw + 1) / 2;
            
            fdm.set_property_value('fcs/elevator-cmd-norm', elev);
            fdm.set_property_value('fcs/aileron-cmd-norm', ail);
            fdm.set_property_value('fcs/throttle-cmd-norm', thr);
            fdm.run();
            
            alt_rl(i) = double(fdm.get_property_value('position/h-sl-ft'));
        end
        err_rl = alt_rl(settle:end) - 4000;
        rms_alt_rl(s, r) = sqrt(mean(err_rl.^2));
        max_alt_rl(s, r) = max(abs(err_rl));
    end
    disp([' PID Mean Sev ' num2str(sev) ': ' num2str(mean(rms_alt_pid(s,:)))]);
    disp([' RL Mean Sev ' num2str(sev) ': ' num2str(mean(rms_alt_rl(s,:)))]);
end

% --- SAVE WORKSPACE DATA ---
disp('Saving raw monte-carlo data to monte_carlo_raw_data.mat...');
save('monte_carlo_raw_data.mat', 'rms_alt_pid', 'rms_alt_rl', 'max_alt_pid', 'max_alt_rl', 'severities', 'N_RUNS');

% --- PLOTTING MATHEMATICAL BOXPLOTS ---
figure('Position', [50, 100, 1200, 500]); set(gcf, 'color', 'w');

% Correctly align the categorical mapping (Flattening unrolls column-wise)
% Since data is (n_sev x N_RUNS), flattening it creates sequences of 1..5
sev_col = repmat((1:5)', N_RUNS, 1);
x_data = [sev_col; sev_col]; 
g_data = [repmat(categorical({'PID'}), numel(rms_alt_pid), 1); repmat(categorical({'RL Agent'}), numel(rms_alt_rl), 1)];

% RMS Boxplot (Left)
subplot(1,2,1);
y_rms = [rms_alt_pid(:); rms_alt_rl(:)];
% Draw connective trendlines
plot(1:5, mean(rms_alt_pid, 2), 'Color', [0.33, 0.49, 0.74, 0.5], 'LineWidth', 2); hold on;
plot(1:5, mean(rms_alt_rl, 2), 'Color', [0.42, 0.71, 0.43, 0.5], 'LineWidth', 2);
% Draw boxes
boxchart(x_data, y_rms, 'GroupByColor', g_data, 'LineWidth', 1.5, 'MarkerStyle', 'o');
colororder([0.33, 0.49, 0.74; 0.42, 0.71, 0.43]);
ylabel('RMS Altitude Error (ft)');
xlabel('Turbulence Severity');
title('RMS Altitude Error Comparison');
legend('PID', 'PPO Agent', 'Location', 'northwest');
grid on;
set(gca, 'XTick', 1:5, 'XTickLabel', {'Light', 'Light-Moderate', 'Moderate', 'Moderate-Severe', 'Severe'});

% Peak Deviation Boxplot (Right)
subplot(1,2,2);
y_max = [max_alt_pid(:); max_alt_rl(:)];
% Draw connective trendlines
plot(1:5, max(max_alt_pid, [], 2), 'Color', [0.33, 0.49, 0.74, 0.5], 'LineWidth', 2); hold on;
plot(1:5, max(max_alt_rl, [], 2), 'Color', [0.42, 0.71, 0.43, 0.5], 'LineWidth', 2);
% Draw boxes
boxchart(x_data, y_max, 'GroupByColor', g_data, 'LineWidth', 1.5, 'MarkerStyle', 'o');
colororder([0.33, 0.49, 0.74; 0.42, 0.71, 0.43]);
ylabel('Peak Altitude Deviation (ft)');
xlabel('Turbulence Severity');
title('Peak Altitude Deviation Comparison');
legend('PID', 'PPO Agent', 'Location', 'northwest');
grid on;
set(gca, 'XTick', 1:5, 'XTickLabel', {'Light', 'Light-Moderate', 'Moderate', 'Moderate-Severe', 'Severe'});

sgtitle(['Monte Carlo Sweep (' num2str(N_RUNS) ' runs per severity level)']);

disp('Generating and saving final chart...');
saveas(gcf, 'severity_monte_carlo_boxplots.png');
disp('Done!');
