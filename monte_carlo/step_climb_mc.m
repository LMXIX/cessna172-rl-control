% physicstest_mc.m
% 10-Run Parametric Monte Carlo for 500ft Step-Climb
% Varies the Center of Gravity (CG) by randomizing the Rear Passenger Payload 
% between 0 lbs (Empty) and 390 lbs (Max capacity aft-CG).

clear all; clc; close all;

t_end = 120;
dt = 0.02;
n_steps = floor(t_end / dt);
t_vec = linspace(0, t_end, n_steps);

% Create the 500ft step command (at 10 seconds)
cmd_h = 4000 * ones(1, n_steps);
cmd_h(t_vec >= 10) = 4500; 

N_RUNS = 10;
base_seed = 3000;

% Storage Arrays
alt_pid_all = zeros(N_RUNS, n_steps);
spd_pid_all = zeros(N_RUNS, n_steps);
theta_pid_all = zeros(N_RUNS, n_steps);
elev_pid_all = zeros(N_RUNS, n_steps);

alt_rl_all = zeros(N_RUNS, n_steps);
spd_rl_all = zeros(N_RUNS, n_steps);
theta_rl_all = zeros(N_RUNS, n_steps);
elev_rl_all = zeros(N_RUNS, n_steps);

%% Load v2 Agent
disp('Loading Finetuned v2 Agent...');
loaded = load('curriculum_models/Phase4_finetune/Final_v2_Finetuned.mat');
fields = fieldnames(loaded);
agent = loaded.(fields{1});
actor = getActor(agent);
disp('Agent loaded.');

%% Helper Init Function
function fdm = init_jsbsim_local()
    try
        py.importlib.import_module('jsbsim');
    catch
        pyenv('Version', '/Library/Frameworks/Python.framework/Versions/3.12/bin/python3');
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
    
    % ZERO WIND FOR PARAMETRIC EVALUATION
    fdm.set_property_value('atmosphere/turb-type', 4);
    fdm.set_property_value('atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps', 0);
    fdm.set_property_value('atmosphere/turbulence/milspec/severity', 0);
    
    fdm.set_property_value('fcs/throttle-cmd-norm', 0.80);
    for k = 1:25, fdm.run(); end
end

%% Run the Parametric Monte Carlo loops
disp('=== Running Parametric Mass/CG Monte Carlo ===');

kp_h = 0.008; ki_h = 0.002; kd_h = 0.018;
kp_theta = 2.5; kd_q = 0.8;
kp_psi = 0.5; kp_phi = 1.5; ki_phi = 0.1; kd_p = 0.2;
POLARITY = -1.0;

for r = 1:N_RUNS
    rng(base_seed + r);
    % Randomize passenger payload between 0 and 390 lbs to shift CG significantly
    payload_pax = rand * 390; 
    disp(['  > Run ' num2str(r) '/' num2str(N_RUNS) ' | Rear Passenger Weight: ' num2str(round(payload_pax)) ' lbs']);
    
    % --- PID ---
    fdm_pid = init_jsbsim_local();
    % pointmass [2] and [3] are rear passenger seats in JSBSim C172
    fdm_pid.set_property_value('inertia/pointmass-weight-lbs[2]', payload_pax/2);
    fdm_pid.set_property_value('inertia/pointmass-weight-lbs[3]', payload_pax/2);
    
    int_h = 0; int_phi = 0;
    for i = 1:n_steps
        target_alt = cmd_h(i);
        
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
        
        err_h = target_alt - h;
        int_h = max(min(int_h + (err_h * dt), 20), -20);
        theta_cmd = max(min((kp_h*err_h) + (ki_h*int_h) - (kd_h*h_dot), 0.087), -0.15);
        elev = max(min(POLARITY * (kp_theta*(theta_cmd-theta) - kd_q*q), 1), -1);
        fdm_pid.set_property_value('fcs/elevator-cmd-norm', elev);
        
        err_psi = 0 - psi;
        phi_cmd = max(min(kp_psi * err_psi, 0.35), -0.35);
        err_phi = phi_cmd - phi;
        int_phi = max(min(int_phi + (err_phi * dt), 0.5), -0.5);
        ail = max(min((kp_phi*err_phi) + (ki_phi*int_phi) - (kd_p*p), 1), -1);
        fdm_pid.set_property_value('fcs/aileron-cmd-norm', ail);
        
        fdm_pid.run();
        alt_pid_all(r, i) = h;
        spd_pid_all(r, i) = v;
        theta_pid_all(r, i) = rad2deg(theta);
        elev_pid_all(r, i) = elev;
    end
    
    % --- RL ---
    fdm_rl = init_jsbsim_local(); % initialize natively
    fdm_rl.set_property_value('inertia/pointmass-weight-lbs[2]', payload_pax/2);
    fdm_rl.set_property_value('inertia/pointmass-weight-lbs[3]', payload_pax/2);
    
    alpha_filter = 0.05; prev_elev = 0;
    
    for i = 1:n_steps
        target_alt = cmd_h(i);
        
        h = double(fdm_rl.get_property_value('position/h-sl-ft'));
        h_dot = double(fdm_rl.get_property_value('velocities/h-dot-fps'));
        th = double(fdm_rl.get_property_value('attitude/theta-rad'));
        q = double(fdm_rl.get_property_value('velocities/q-rad_sec'));
        phi = double(fdm_rl.get_property_value('attitude/phi-rad'));
        p = double(fdm_rl.get_property_value('velocities/p-rad_sec'));
        v = double(fdm_rl.get_property_value('velocities/vc-kts'));
        elev_pos = double(fdm_rl.get_property_value('fcs/elevator-cmd-norm'));
        
        err_h_c = h - target_alt;
        err_v_c = v - 100;
        obs = [err_h_c/1000; h_dot/100; th; q; phi; p; err_v_c/10; elev_pos];
        
        action = evaluate(actor, {obs});
        act_num = cell2mat(action);
        
        elev_raw = max(min(act_num(1), 1), -1);
        elev = alpha_filter * elev_raw + (1 - alpha_filter) * prev_elev;
        prev_elev = elev;
        ail = max(min(act_num(2), 1), -1);
        thr_raw = max(min(act_num(3), 1), -1);
        thr = (thr_raw + 1) / 2;
        
        fdm_rl.set_property_value('fcs/elevator-cmd-norm', elev);
        fdm_rl.set_property_value('fcs/aileron-cmd-norm', ail);
        fdm_rl.set_property_value('fcs/throttle-cmd-norm', thr);
        fdm_rl.run();
        
        alt_rl_all(r, i) = h;
        spd_rl_all(r, i) = v;
        theta_rl_all(r, i) = rad2deg(th);
        elev_rl_all(r, i) = elev;
    end
end
disp('Simulations complete. Plotting...');

%% Compute Averages
mean_alt_pid = mean(alt_pid_all, 1);
mean_spd_pid = mean(spd_pid_all, 1);
mean_theta_pid = mean(theta_pid_all, 1);
mean_elev_pid = mean(elev_pid_all, 1);

mean_alt_rl = mean(alt_rl_all, 1);
mean_spd_rl = mean(spd_rl_all, 1);
mean_theta_rl = mean(theta_rl_all, 1);
mean_elev_rl = mean(elev_rl_all, 1);

%% Academic Plotting Routine
fig = figure('Position', [50, 50, 1000, 700], 'Color', 'w');
color_pid = [0.33, 0.49, 0.74];
color_rl  = [0.42, 0.71, 0.43];
color_pid_trans = [0.33, 0.49, 0.74, 0.25]; 
color_rl_trans  = [0.42, 0.71, 0.43, 0.25]; 

% 1. ALTITUDE
subplot(2,2,1);
for r = 1:N_RUNS
    plot(t_vec, alt_pid_all(r,:), 'Color', color_pid_trans); hold on;
    plot(t_vec, alt_rl_all(r,:), 'Color', color_rl_trans);
end
p1 = plot(t_vec, mean_alt_pid, 'Color', color_pid, 'LineWidth', 2);
p2 = plot(t_vec, mean_alt_rl, 'Color', color_rl, 'LineWidth', 2);
plot(t_vec, cmd_h, 'k--', 'LineWidth', 1.2);
title('Altitude Response');
ylabel('Altitude (ft)'); xlabel('Time (s)'); grid on;
legend([p1, p2], {'Mean PID', 'Mean RL'}, 'Location', 'southeast');

% 2. PITCH
subplot(2,2,2);
for r = 1:N_RUNS
    plot(t_vec, theta_pid_all(r,:), 'Color', color_pid_trans); hold on;
    plot(t_vec, theta_rl_all(r,:), 'Color', color_rl_trans);
end
plot(t_vec, mean_theta_pid, 'Color', color_pid, 'LineWidth', 2);
plot(t_vec, mean_theta_rl, 'Color', color_rl, 'LineWidth', 2);
title('Pitch Attitude');
ylabel('Pitch Angle (\circ)'); xlabel('Time (s)'); grid on;

% 3. AIRSPEED
subplot(2,2,3);
for r = 1:N_RUNS
    plot(t_vec, spd_pid_all(r,:), 'Color', color_pid_trans); hold on;
    plot(t_vec, spd_rl_all(r,:), 'Color', color_rl_trans);
end
plot(t_vec, mean_spd_pid, 'Color', color_pid, 'LineWidth', 2);
plot(t_vec, mean_spd_rl, 'Color', color_rl, 'LineWidth', 2);
title('Airspeed Response');
ylabel('Airspeed (kts)'); xlabel('Time (s)'); grid on;

% 4. ELEVATOR
subplot(2,2,4);
for r = 1:N_RUNS
    plot(t_vec, elev_pid_all(r,:), 'Color', color_pid_trans); hold on;
    plot(t_vec, elev_rl_all(r,:), 'Color', color_rl_trans);
end
plot(t_vec, mean_elev_pid, 'Color', color_pid, 'LineWidth', 2);
plot(t_vec, mean_elev_rl, 'Color', color_rl, 'LineWidth', 2);
title('Elevator Actuation');
ylabel('Elevator (norm)'); xlabel('Time (s)'); grid on;

sgtitle('10-Run Parametric Mass/CG Monte Carlo (500ft Step-Climb)');

save('physicstest_mc_data.mat', 'alt_pid_all', 'alt_rl_all', 'spd_pid_all', 'spd_rl_all');
disp('Saving image...');
saveas(gcf, 'physicstest_mc.png');
disp('Done!');
