% scenIcing_sweep.m
% Parametric Sweep of Elevator Authority Degradation
% Varies elevator effectiveness from 80% down to 15%
% Same 25fps downdraft at t=60-75s for every run
% Produces a survivability envelope comparing PID vs RL

clear all; clc; close all;

%% SWEEP PARAMETERS
authority_levels = [0.80, 0.70, 0.60, 0.50, 0.40];
N_AUTH = length(authority_levels);

ICING_START  = 20;
ICING_END    = 40;
GUST_START   = 60;
GUST_END     = 75;
GUST_MAG_FPS = 25;

t_end = 120; dt = 0.02;
n_steps = floor(t_end / dt);
t_vec = (1:n_steps)' * dt;

%% Load Agent
disp('Loading Finetuned v2 Agent...');
loaded = load('curriculum_models/Phase4_finetune/Final_v2_Finetuned.mat');
fields = fieldnames(loaded);
agent = loaded.(fields{1});
actor = getActor(agent);
disp('  Agent loaded.');

%% PID Gains
kp_h = 0.008; ki_h = 0.002; kd_h = 0.018;
kp_theta = 2.5; kd_q = 0.8;
kp_psi = 0.5; kp_phi = 1.5; ki_phi = 0.1; kd_p = 0.2;
POLARITY = -1.0;

%% Helper: Init JSBSim
function fdm = init_jsbsim_calm()
    try py.importlib.import_module('jsbsim');
    catch
        pyenv('Version','/Library/Frameworks/Python.framework/Versions/3.12/bin/python3');
        py.importlib.import_module('jsbsim');
    end
    root = '/Users/lee-michaelcookhorn/Documents/jsbsim/jsbsim-master';
    fdm = py.jsbsim.FGFDMExec(root);
    fdm.set_aircraft_path([root '/aircraft']);
    fdm.set_engine_path([root '/engine']);
    fdm.set_systems_path([root '/systems']);
    fdm.load_script(py.str([root '/scripts/c172_cruise_test.xml']));
    fdm.set_dt(0.02);
    fdm.set_property_value('ic/h-sl-ft', 4000);
    fdm.set_property_value('ic/vc-kts', 100);
    fdm.run_ic();
    fdm.set_property_value('propulsion/tank[0]/contents-lbs', 150);
    fdm.set_property_value('propulsion/tank[1]/contents-lbs', 150);
    fdm.set_property_value('fcs/mixture-cmd-norm', 0.85);
    fdm.set_property_value('propulsion/magneto_cmd', 3);
    fdm.set_property_value('propulsion/starter_cmd', 1);
    fdm.set_property_value('propulsion/engine[0]/set-running', 1);
    fdm.set_property_value('atmosphere/turb-type', 0);
    fdm.set_property_value('atmosphere/wind-north-fps', 0);
    fdm.set_property_value('atmosphere/wind-east-fps', 0);
    fdm.set_property_value('atmosphere/wind-down-fps', 0);
    fdm.set_property_value('fcs/throttle-cmd-norm', 0.80);
    for k = 1:25, fdm.run(); end
end

%% Authority helper
function auth = get_authority(t, icing_start, icing_end, min_auth)
    if t < icing_start,        auth = 1.0;
    elseif t < icing_end,      auth = 1.0 - ((t-icing_start)/(icing_end-icing_start)) * (1.0-min_auth);
    else,                      auth = min_auth;
    end
end

%% Storage for sweep results
final_alt_pid  = zeros(N_AUTH, 1);  % altitude at t=120
final_alt_rl   = zeros(N_AUTH, 1);
max_dev_pid    = zeros(N_AUTH, 1);  % max |alt - 4000|
max_dev_rl     = zeros(N_AUTH, 1);
min_spd_pid    = zeros(N_AUTH, 1);
min_spd_rl     = zeros(N_AUTH, 1);
rms_pid        = zeros(N_AUTH, 1);
rms_rl         = zeros(N_AUTH, 1);

%% RUN THE SWEEP
for a = 1:N_AUTH
    min_auth = authority_levels(a);
    disp(['=== Authority: ' num2str(min_auth*100) '% ===']);
    
    % --- PID ---
    fdm_pid = init_jsbsim_calm();
    int_h = 0; int_phi = 0;
    alt_pid_run = zeros(n_steps,1);
    spd_pid_run = zeros(n_steps,1);
    
    for i = 1:n_steps
        t = t_vec(i);
        authority = get_authority(t, ICING_START, ICING_END, min_auth);
        
        if t >= GUST_START && t < GUST_END
            fdm_pid.set_property_value('atmosphere/wind-down-fps', GUST_MAG_FPS);
        else
            fdm_pid.set_property_value('atmosphere/wind-down-fps', 0);
        end
        
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
        fdm_pid.set_property_value('fcs/elevator-cmd-norm', elev * authority);
        
        err_psi = 0 - psi;
        phi_cmd = max(min(kp_psi * err_psi, 0.35), -0.35);
        err_phi = phi_cmd - phi;
        int_phi = max(min(int_phi + (err_phi * dt), 0.5), -0.5);
        ail = max(min((kp_phi*err_phi) + (ki_phi*int_phi) - (kd_p*p), 1), -1);
        fdm_pid.set_property_value('fcs/aileron-cmd-norm', ail);
        
        fdm_pid.run();
        alt_pid_run(i) = h;
        spd_pid_run(i) = v;
    end
    
    % --- RL ---
    fdm_rl = init_jsbsim_calm();
    alpha_filter = 0.05; prev_elev = 0;
    alt_rl_run = zeros(n_steps,1);
    spd_rl_run = zeros(n_steps,1);
    
    for i = 1:n_steps
        t = t_vec(i);
        authority = get_authority(t, ICING_START, ICING_END, min_auth);
        
        if t >= GUST_START && t < GUST_END
            fdm_rl.set_property_value('atmosphere/wind-down-fps', GUST_MAG_FPS);
        else
            fdm_rl.set_property_value('atmosphere/wind-down-fps', 0);
        end
        
        h = double(fdm_rl.get_property_value('position/h-sl-ft'));
        h_dot = double(fdm_rl.get_property_value('velocities/h-dot-fps'));
        th = double(fdm_rl.get_property_value('attitude/theta-rad'));
        q = double(fdm_rl.get_property_value('velocities/q-rad_sec'));
        phi = double(fdm_rl.get_property_value('attitude/phi-rad'));
        p = double(fdm_rl.get_property_value('velocities/p-rad_sec'));
        v = double(fdm_rl.get_property_value('velocities/vc-kts'));
        elev_pos = double(fdm_rl.get_property_value('fcs/elevator-cmd-norm'));
        
        err_h_c = h - 4000;
        err_v_c = v - 100;
        obs = [err_h_c/1000; h_dot/100; th; q; phi; p; err_v_c/10; elev_pos];
        
        action = evaluate(actor, {obs});
        act = cell2mat(action);
        
        elev_raw = max(min(act(1), 1), -1);
        elev_cmd = alpha_filter * elev_raw + (1 - alpha_filter) * prev_elev;
        prev_elev = elev_cmd;
        fdm_rl.set_property_value('fcs/elevator-cmd-norm', elev_cmd * authority);
        
        ail = max(min(act(2), 1), -1);
        fdm_rl.set_property_value('fcs/aileron-cmd-norm', ail);
        
        thr_raw = max(min(act(3), 1), -1);
        fdm_rl.set_property_value('fcs/throttle-cmd-norm', (thr_raw + 1) / 2);
        
        fdm_rl.run();
        alt_rl_run(i) = h;
        spd_rl_run(i) = v;
    end
    
    % Compute metrics (skip first 20s for baseline)
    settle = floor(20/dt);
    err_pid = alt_pid_run(settle:end) - 4000;
    err_rl  = alt_rl_run(settle:end) - 4000;
    
    final_alt_pid(a) = alt_pid_run(end);
    final_alt_rl(a)  = alt_rl_run(end);
    max_dev_pid(a)   = max(abs(err_pid));
    max_dev_rl(a)    = max(abs(err_rl));
    min_spd_pid(a)   = min(spd_pid_run);
    min_spd_rl(a)    = min(spd_rl_run);
    rms_pid(a)       = sqrt(mean(err_pid.^2));
    rms_rl(a)        = sqrt(mean(err_rl.^2));
    
    fprintf('  PID: final alt=%.0f ft, max dev=%.0f ft, min spd=%.1f kts\n', ...
        final_alt_pid(a), max_dev_pid(a), min_spd_pid(a));
    fprintf('  RL:  final alt=%.0f ft, max dev=%.0f ft, min spd=%.1f kts\n', ...
        final_alt_rl(a), max_dev_rl(a), min_spd_rl(a));
end

%% PLOTTING
color_pid = [0.33, 0.49, 0.74];
color_rl  = [0.42, 0.71, 0.43];
auth_pct = authority_levels * 100;

fig = figure('Position', [50, 50, 1100, 800], 'Color', 'w');

% 1. Final Altitude at t=120s
subplot(2,2,1);
plot(auth_pct, final_alt_pid, '-o', 'Color', color_pid, 'LineWidth', 2, 'MarkerFaceColor', color_pid); hold on;
plot(auth_pct, final_alt_rl, '-s', 'Color', color_rl, 'LineWidth', 2, 'MarkerFaceColor', color_rl);
yline(4000, 'k--', 'Target', 'LineWidth', 0.8);
set(gca, 'XDir', 'reverse');
title('Final Altitude at t=120s');
xlabel('Elevator Authority (%)'); ylabel('Altitude (ft)'); grid on;
legend('PID', 'RL', 'Location', 'best');

% 2. Peak Altitude Deviation
subplot(2,2,2);
plot(auth_pct, max_dev_pid, '-o', 'Color', color_pid, 'LineWidth', 2, 'MarkerFaceColor', color_pid); hold on;
plot(auth_pct, max_dev_rl, '-s', 'Color', color_rl, 'LineWidth', 2, 'MarkerFaceColor', color_rl);
set(gca, 'XDir', 'reverse');
title('Peak Altitude Deviation');
xlabel('Elevator Authority (%)'); ylabel('|Deviation| (ft)'); grid on;
legend('PID', 'RL', 'Location', 'best');

% 3. RMS Altitude Error
subplot(2,2,3);
plot(auth_pct, rms_pid, '-o', 'Color', color_pid, 'LineWidth', 2, 'MarkerFaceColor', color_pid); hold on;
plot(auth_pct, rms_rl, '-s', 'Color', color_rl, 'LineWidth', 2, 'MarkerFaceColor', color_rl);
set(gca, 'XDir', 'reverse');
title('RMS Altitude Error');
xlabel('Elevator Authority (%)'); ylabel('RMS Error (ft)'); grid on;
legend('PID', 'RL', 'Location', 'best');

% 4. Minimum Airspeed
subplot(2,2,4);
plot(auth_pct, min_spd_pid, '-o', 'Color', color_pid, 'LineWidth', 2, 'MarkerFaceColor', color_pid); hold on;
plot(auth_pct, min_spd_rl, '-s', 'Color', color_rl, 'LineWidth', 2, 'MarkerFaceColor', color_rl);
yline(55, 'r--', 'Stall', 'LineWidth', 1);
set(gca, 'XDir', 'reverse');
title('Minimum Airspeed');
xlabel('Elevator Authority (%)'); ylabel('Min Airspeed (kts)'); grid on;
legend('PID', 'RL', 'Location', 'best');

sgtitle('Elevator Icing Survivability Envelope (25 fps downdraft at t=60s)');

exportgraphics(fig, 'scenIcing_sweep.png', 'Resolution', 300);
save('scenIcing_sweep_data.mat', 'authority_levels', 'final_alt_pid', 'final_alt_rl', ...
    'max_dev_pid', 'max_dev_rl', 'min_spd_pid', 'min_spd_rl', 'rms_pid', 'rms_rl');
disp('Saved: scenIcing_sweep.png and scenIcing_sweep_data.mat');
disp('Done.');
