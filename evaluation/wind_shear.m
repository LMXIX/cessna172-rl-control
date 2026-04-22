% Wind Shear Scenario
% 20kt headwind to 20kt tailwind at t=30s
% v2 Agent vs Full PID — 120s
% Direct JSBSim control

clear all; clc; close all;

t_end = 120; dt = 0.02;
n_steps = floor(t_end / dt);
t_vec = (1:n_steps)' * dt;
SHEAR_TIME = 30;  % Wind reversal at t=30s

%% Load Fine-Tuned v2 Agent
disp('Loading Fine-Tuned v2 Agent...');
loaded = load('curriculum_models/Phase4_finetune/Final_v2_Finetuned.mat');
fields = fieldnames(loaded);
agent = loaded.(fields{1});
actor = getActor(agent);
disp('  Agent loaded.');

%% PID Gains (Full PID with integral)
kp_h = 0.008; ki_h = 0.002; kd_h = 0.018;
kp_theta = 2.5; kd_q = 0.8;
kp_psi = 0.5; kp_phi = 1.5; ki_phi = 0.1; kd_p = 0.2;
POLARITY = -1.0;

%% Helper: Init JSBSim (no turbulence, headwind start)
function fdm = init_jsbsim_wind()
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
    
    % No turbulence — clean wind shear only
    fdm.set_property_value('atmosphere/turb-type', 0);
    % Start with 20kt headwind (~34 fps)
    fdm.set_property_value('atmosphere/wind-north-fps', -34);
    
    fdm.set_property_value('fcs/throttle-cmd-norm', 0.80);
    for k = 1:25, fdm.run(); end
    fdm.set_property_value('propulsion/tank[0]/contents-lbs', 130);
    fdm.set_property_value('propulsion/tank[1]/contents-lbs', 130);
end

disp('START PID');
fdm_pid = init_jsbsim_wind();

alt_pid   = zeros(n_steps, 1);
spd_pid   = zeros(n_steps, 1);
elev_pid  = zeros(n_steps, 1);
theta_pid = zeros(n_steps, 1);
wind_rec  = zeros(n_steps, 1);
int_h = 0; int_phi_pid = 0;

for i = 1:n_steps
    % WIND SHEAR: 20kt headwind → 20kt tailwind at t=30s
    % Plus 15fps downdraft during transition (microburst)
    if t_vec(i) >= SHEAR_TIME && t_vec(i) < SHEAR_TIME + 20
        fdm_pid.set_property_value('atmosphere/wind-north-fps', 34);  % tailwind
        fdm_pid.set_property_value('atmosphere/wind-down-fps', 15);   % downdraft
        wind_rec(i) = 34;
    elseif t_vec(i) >= SHEAR_TIME + 20
        fdm_pid.set_property_value('atmosphere/wind-north-fps', 34);  % tailwind persists
        fdm_pid.set_property_value('atmosphere/wind-down-fps', 0);    % downdraft ends
        wind_rec(i) = 34;
    else
        wind_rec(i) = -34;
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
    fdm_pid.set_property_value('fcs/elevator-cmd-norm', elev);
    
    err_psi = 0 - psi;
    phi_cmd = max(min(kp_psi * err_psi, 0.35), -0.35);
    err_phi = phi_cmd - phi;
    int_phi_pid = max(min(int_phi_pid + (err_phi * dt), 0.5), -0.5);
    ail = max(min((kp_phi*err_phi) + (ki_phi*int_phi_pid) - (kd_p*p), 1), -1);
    fdm_pid.set_property_value('fcs/aileron-cmd-norm', ail);
    
    fdm_pid.run();
    alt_pid(i) = h; spd_pid(i) = v; elev_pid(i) = elev; theta_pid(i) = rad2deg(theta);
end
disp('  PID run complete.');

fdm_rl = init_jsbsim_wind();

alt_rl   = zeros(n_steps, 1);
spd_rl   = zeros(n_steps, 1);
elev_rl  = zeros(n_steps, 1);
theta_rl = zeros(n_steps, 1);

alpha_filter = 0.05;
prev_elev = 0;

for i = 1:n_steps
    % WIND SHEAR (same timing and profile)
    if t_vec(i) >= SHEAR_TIME && t_vec(i) < SHEAR_TIME + 20
        fdm_rl.set_property_value('atmosphere/wind-north-fps', 34);
        fdm_rl.set_property_value('atmosphere/wind-down-fps', 15);
    elseif t_vec(i) >= SHEAR_TIME + 20
        fdm_rl.set_property_value('atmosphere/wind-north-fps', 34);
        fdm_rl.set_property_value('atmosphere/wind-down-fps', 0);
    end
    
    h = double(fdm_rl.get_property_value('position/h-sl-ft'));
    h_dot = double(fdm_rl.get_property_value('velocities/h-dot-fps'));
    th = double(fdm_rl.get_property_value('attitude/theta-rad'));
    q = double(fdm_rl.get_property_value('velocities/q-rad_sec'));
    phi = double(fdm_rl.get_property_value('attitude/phi-rad'));
    p = double(fdm_rl.get_property_value('velocities/p-rad_sec'));
    v_kts = double(fdm_rl.get_property_value('velocities/vc-kts'));
    elev_pos = double(fdm_rl.get_property_value('fcs/elevator-cmd-norm'));
    
    err_h = h - 4000;
    err_v = v_kts - 100;
    obs = [err_h/1000; h_dot/100; th; q; phi; p; err_v/10; elev_pos];
    
    action = evaluate(actor, {obs});
    act = cell2mat(action);
    agent_elev = double(act(1));
    agent_ail  = double(act(2));
    agent_thr  = double(act(3));
    
    filtered_elev = (alpha_filter * agent_elev) + ((1 - alpha_filter) * prev_elev);
    actual_elev = max(min(filtered_elev, 1), -1);
    actual_ail  = max(min(agent_ail, 1), -1);
    actual_thr  = max(min(agent_thr, 1), 0);
    prev_elev = actual_elev;
    
    fdm_rl.set_property_value('fcs/elevator-cmd-norm', actual_elev);
    fdm_rl.set_property_value('fcs/aileron-cmd-norm', actual_ail);
    fdm_rl.set_property_value('fcs/throttle-cmd-norm', actual_thr);
    fdm_rl.run();
    
    alt_rl(i) = h; spd_rl(i) = v_kts; elev_rl(i) = actual_elev; theta_rl(i) = rad2deg(th);
end
disp('  RL Agent run complete (full 120s).');

% calculate metrics and print
disp('--- wind shear results ---');
disp(['PID alt err: ' num2str(rms_alt_pid) ' | max dev: ' num2str(max_dev_pid)]);
disp(['RL alt err: ' num2str(rms_alt_rl) ' | max dev: ' num2str(max_dev_rl)]);
disp(['PID recovery (s): ' num2str(recov_pid)]);
disp(['RL recovery (s): ' num2str(recov_rl)]);

% 2x2 combined plots
% std academic colors
color_pid = [0.33, 0.49, 0.74];
color_rl  = [0.42, 0.71, 0.43];

fig = figure('Position', [50, 50, 1100, 850]); set(gcf, 'color', 'w');

subplot(2,2,1);
plot(t_vec, alt_pid, 'Color', color_pid); hold on;
plot(t_vec, alt_rl, 'Color', color_rl);
yline(4000, 'k--');
xline(SHEAR_TIME, 'k:');
title('Altitude');
legend('PID', 'RL', 'Target', 'Location', 'southeast');
grid on;
subplot(2,2,2);
plot(t_vec, spd_pid, 'Color', color_pid); hold on;
plot(t_vec, spd_rl, 'Color', color_rl);
yline(100, 'k--');
xline(SHEAR_TIME, 'k:');
title('Airspeed');
legend('PID', 'RL', 'Target', 'Location', 'southeast');
grid on;
subplot(2,2,3);
plot(t_vec, theta_pid, 'Color', color_pid); hold on;
plot(t_vec, theta_rl, 'Color', color_rl);
yline(0, 'k--');
xline(SHEAR_TIME, 'k:');
title('Pitch');
legend('PID', 'RL', 'Location', 'northeast');
grid on;
subplot(2,2,4);
wind_kts = wind_rec * 0.5924;
area(t_vec, wind_kts, 'FaceColor', [0.85 0.85 0.85]);
hold on;
yline(0, 'k--');
xline(SHEAR_TIME, 'k:');
title('Wind');
legend('Wind Profile', 'Location', 'northeast');

sgtitle('Wind Shear Evaluation');

exportgraphics(fig, 'windshear_combined_4panel.png', 'Resolution', 300);
fprintf('\nSaved: windshear_combined_4panel.png (300 DPI)\n');
disp('Done.');
