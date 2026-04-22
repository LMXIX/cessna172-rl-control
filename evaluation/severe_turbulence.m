% Detailed Severity 5 Comparison
% v2 Agent vs Full PID — 120s @ Severity 5

clear all; clc; close all;

t_end = 600; dt = 0.02;
n_steps = floor(t_end / dt);
t_vec = (1:n_steps)' * dt;
SEVERITY = 5;
SEED = 12345;

%% Load v2 Agent
disp('Loading v2 Agent...');
loaded = load('curriculum_models/Phase4_finetune/Final_v2_Finetuned.mat');
fields = fieldnames(loaded);
agent = loaded.(fields{1});
actor = getActor(agent);
disp('  Fine-tuned agent loaded.');

%% PID Gains (Full PID with integral)
kp_h = 0.008; ki_h = 0.002; kd_h = 0.018;
kp_theta = 2.5; kd_q = 0.8;
kp_psi = 0.5; kp_phi = 1.5; ki_phi = 0.1; kd_p = 0.2;
POLARITY = -1.0;

% setup fn
function fdm = init_jsbsim(severity, seed)
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
    
    fdm.set_property_value('atmosphere/turb-type', 4);
    fdm.set_property_value('atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps', 20);
    fdm.set_property_value('atmosphere/turbulence/milspec/severity', severity);
    fdm.set_property_value('atmosphere/turbulence/milspec/seed', seed);
    
    fdm.set_property_value('fcs/throttle-cmd-norm', 0.80);
    for k = 1:25, fdm.run(); end
    fdm.set_property_value('propulsion/tank[0]/contents-lbs', 130);
    fdm.set_property_value('propulsion/tank[1]/contents-lbs', 130);
end

% do pid
disp(['pid run ' num2str(SEVERITY)]);
fdm_pid = init_jsbsim(SEVERITY, SEED);

alt_pid   = zeros(n_steps, 1);
spd_pid   = zeros(n_steps, 1);
elev_pid  = zeros(n_steps, 1);
theta_pid = zeros(n_steps, 1);
phi_pid   = zeros(n_steps, 1);
wind_n    = zeros(n_steps, 1);
wind_e    = zeros(n_steps, 1);
wind_d    = zeros(n_steps, 1);
int_h = 0; int_phi_pid = 0;

for i = 1:n_steps
    h = double(fdm_pid.get_property_value('position/h-sl-ft'));
    h_dot = double(fdm_pid.get_property_value('velocities/h-dot-fps'));
    v = double(fdm_pid.get_property_value('velocities/vc-kts'));
    theta = double(fdm_pid.get_property_value('attitude/theta-rad'));
    phi = double(fdm_pid.get_property_value('attitude/phi-rad'));
    q = double(fdm_pid.get_property_value('velocities/q-rad_sec'));
    p = double(fdm_pid.get_property_value('velocities/p-rad_sec'));
    psi = double(fdm_pid.get_property_value('attitude/psi-rad'));
    if psi > pi, psi = psi - 2*pi; end
    
    % Record wind (from PID run — same seed)
    wind_n(i) = double(fdm_pid.get_property_value('atmosphere/total-wind-north-fps'));
    wind_e(i) = double(fdm_pid.get_property_value('atmosphere/total-wind-east-fps'));
    wind_d(i) = double(fdm_pid.get_property_value('atmosphere/total-wind-down-fps'));
    
    % Speed hold
    err_v = 100 - v;
    thr_cmd = max(min(0.65 + (0.02 * err_v), 1.0), 0.0);
    fdm_pid.set_property_value('fcs/throttle-cmd-norm', thr_cmd);
    
    % Altitude → Pitch → Elevator
    err_h = 4000 - h;
    int_h = max(min(int_h + (err_h * dt), 20), -20);
    theta_cmd = max(min((kp_h*err_h) + (ki_h*int_h) - (kd_h*h_dot), 0.087), -0.15);
    elev = max(min(POLARITY * (kp_theta*(theta_cmd-theta) - kd_q*q), 1), -1);
    fdm_pid.set_property_value('fcs/elevator-cmd-norm', elev);
    
    % Heading → Roll → Aileron
    err_psi = 0 - psi;
    phi_cmd = max(min(kp_psi * err_psi, 0.35), -0.35);
    err_phi = phi_cmd - phi;
    int_phi_pid = max(min(int_phi_pid + (err_phi * dt), 0.5), -0.5);
    ail = max(min((kp_phi*err_phi) + (ki_phi*int_phi_pid) - (kd_p*p), 1), -1);
    fdm_pid.set_property_value('fcs/aileron-cmd-norm', ail);
    
    fdm_pid.run();
    
    alt_pid(i) = h;
    spd_pid(i) = v;
    elev_pid(i) = elev;
    theta_pid(i) = rad2deg(theta);
    phi_pid(i) = rad2deg(phi);
end
disp('  PID run complete.');

% do rl agent
disp(['rl agent run ' num2str(SEVERITY)]);
fdm_rl = init_jsbsim(SEVERITY, SEED);

alt_rl   = zeros(n_steps, 1);
spd_rl   = zeros(n_steps, 1);
elev_rl  = zeros(n_steps, 1);
theta_rl = zeros(n_steps, 1);
phi_rl   = zeros(n_steps, 1);

alpha_filter = 0.05;  % Agent2851's training value
prev_elev = 0;

for i = 1:n_steps
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
    
    alt_rl(i) = h;
    spd_rl(i) = v_kts;
    elev_rl(i) = actual_elev;
    theta_rl(i) = rad2deg(th);
    phi_rl(i) = rad2deg(phi);
end
disp('  RL Agent run complete (full 120s).');

%% ============================================
%  COMPUTE METRICS (skip first 5s)
%  ============================================
settle = floor(5 / dt);
idx = settle:n_steps;

err_alt_pid = alt_pid(idx) - 4000;
err_alt_rl  = alt_rl(idx)  - 4000;

rms_alt_pid = sqrt(mean(err_alt_pid.^2));
rms_alt_rl  = sqrt(mean(err_alt_rl.^2));
max_dev_pid = max(abs(err_alt_pid));
max_dev_rl  = max(abs(err_alt_rl));
mean_alt_pid = mean(err_alt_pid);
mean_alt_rl  = mean(err_alt_rl);
std_alt_pid  = std(err_alt_pid);
std_alt_rl   = std(err_alt_rl);

min_spd_pid = min(spd_pid(idx));
min_spd_rl  = min(spd_rl(idx));
max_spd_pid = max(spd_pid(idx));
max_spd_rl  = max(spd_rl(idx));
rms_spd_pid = sqrt(mean((spd_pid(idx)-100).^2));
rms_spd_rl  = sqrt(mean((spd_rl(idx)-100).^2));

elev_rate_pid = diff(elev_pid) / dt;
elev_rate_rl  = diff(elev_rl) / dt;
rms_elev_rate_pid = sqrt(mean(elev_rate_pid.^2));
rms_elev_rate_rl  = sqrt(mean(elev_rate_rl.^2));

pct_50ft_pid = 100 * sum(abs(err_alt_pid) < 50) / length(err_alt_pid);
pct_50ft_rl  = 100 * sum(abs(err_alt_rl) < 50) / length(err_alt_rl);
pct_20ft_pid = 100 * sum(abs(err_alt_pid) < 20) / length(err_alt_pid);
pct_20ft_rl  = 100 * sum(abs(err_alt_rl) < 20) / length(err_alt_rl);

[~, worst_pid_idx] = max(abs(err_alt_pid));
worst_pid_time = t_vec(idx(worst_pid_idx));
worst_pid_alt  = alt_pid(idx(worst_pid_idx));
[~, worst_rl_idx] = max(abs(err_alt_rl));
worst_rl_time = t_vec(idx(worst_rl_idx));
worst_rl_alt  = alt_rl(idx(worst_rl_idx));

% print out results
disp('--- results ---');
disp(['PID alt err: ' num2str(rms_alt_pid) ' | max dev: ' num2str(max_dev_pid)]);
disp(['RL alt err: ' num2str(rms_alt_rl) ' | max dev: ' num2str(max_dev_rl)]);
disp(['PID spd err: ' num2str(rms_spd_pid)]);
disp(['RL spd err: ' num2str(rms_spd_rl)]);
disp(['PID 50ft: ' num2str(pct_50ft_pid) '%']);
disp(['RL 50ft: ' num2str(pct_50ft_rl) '%']);
disp(['PID 20ft: ' num2str(pct_20ft_pid) '%']);
disp(['RL 20ft: ' num2str(pct_20ft_rl) '%']);

% individual plots
% std colors
color_pid = [0.33, 0.49, 0.74]; 
color_rl  = [0.42, 0.71, 0.43]; 

% Compute wind magnitude
wind_mag = sqrt(wind_n.^2 + wind_e.^2 + wind_d.^2) * 0.5924; % fps to kts

figure;
plot(t_vec, wind_mag); hold on;
plot(t_vec, wind_d * 0.5924);
title(['Wind Severity ' num2str(SEVERITY)]);
legend('total', 'vert gust');
grid on;
saveas(gcf, sprintf('sev%d_wind.png', SEVERITY));
fprintf('Saved: sev%d_wind.png\n', SEVERITY);

figure;
plot(t_vec, alt_pid, 'Color', color_pid); hold on;
plot(t_vec, alt_rl, 'Color', color_rl);
yline(4000, 'k--');
title('Altitude');
legend('PID', 'RL', 'Target');
grid on;
saveas(gcf, sprintf('sev%d_altitude.png', SEVERITY));
fprintf('Saved: sev%d_altitude.png\n', SEVERITY);

figure;
plot(t_vec, spd_pid, 'Color', color_pid); hold on;
plot(t_vec, spd_rl, 'Color', color_rl);
yline(100, 'k--');
yline(55, 'r--');
title('Airspeed');
legend('PID', 'RL', 'Target');
grid on;
saveas(gcf, sprintf('sev%d_airspeed.png', SEVERITY));
fprintf('Saved: sev%d_airspeed.png\n', SEVERITY);

figure;
plot(t_vec, theta_pid, 'Color', color_pid); hold on;
plot(t_vec, theta_rl, 'Color', color_rl);
yline(0, 'k--');
title('Pitch');
legend('PID', 'RL');
grid on;
saveas(gcf, sprintf('sev%d_pitch.png', SEVERITY));
fprintf('Saved: sev%d_pitch.png\n', SEVERITY);

figure;
plot(t_vec, elev_pid, 'Color', color_pid); hold on;
plot(t_vec, elev_rl, 'Color', color_rl);
title('Elevator');
grid on;
saveas(gcf, sprintf('sev%d_elevator.png', SEVERITY));
fprintf('Saved: sev%d_elevator.png\n', SEVERITY);

disp('All individual figures saved.');

% combined 5 panel big figure
fig_combined = figure('Position', [50, 50, 1200, 800]); set(gcf, 'color', 'w');

% (a) Wind Profile
subplot(2,3,1);
yyaxis left;
plot(t_vec, wind_mag, 'Color', [0.4 0.4 0.4], 'LineWidth', 0.8);
ylabel('Total Wind (kts)');
yyaxis right;
plot(t_vec, wind_d * 0.5924, 'Color', [0.1 0.5 0.1], 'LineWidth', 0.8);
ylabel('Vertical Gust (kts)');
title('Wind Profile');
grid on; xlim([0 t_end]);
xlabel('Time (s)');
set(gca, 'FontSize', 10);

% (b) Altitude
subplot(2,3,2);
plot(t_vec, alt_pid, 'Color', color_pid, 'LineWidth', 1.6); hold on;
plot(t_vec, alt_rl, 'Color', color_rl, 'LineWidth', 1.6);
yline(4000, 'k--', 'LineWidth', 0.8);
ylabel('Altitude (ft)');
title('Altitude Response');
legend('PID', 'PPO Agent', 'Target', 'Location', 'best');
grid on; xlim([0 t_end]);
xlabel('Time (s)');
set(gca, 'FontSize', 10);

% (c) Airspeed
subplot(2,3,3);
plot(t_vec, spd_pid, 'Color', color_pid, 'LineWidth', 1.6); hold on;
plot(t_vec, spd_rl, 'Color', color_rl, 'LineWidth', 1.6);
yline(100, 'k--', 'LineWidth', 0.8);
ylabel('Airspeed (kts)');
title('Airspeed Response');
legend('PID', 'PPO Agent', 'Target', 'Location', 'best');
grid on; xlim([0 t_end]);
xlabel('Time (s)');
set(gca, 'FontSize', 10);

% (d) Pitch Attitude
subplot(2,3,4);
plot(t_vec, theta_pid, 'Color', color_pid, 'LineWidth', 1.4); hold on;
plot(t_vec, theta_rl, 'Color', color_rl, 'LineWidth', 1.4);
yline(0, 'k--', 'LineWidth', 0.5);
ylabel('Pitch Angle (\circ)');
title('Pitch Attitude');
legend('PID', 'PPO Agent', 'Location', 'best');
grid on; xlim([0 t_end]);
xlabel('Time (s)');
set(gca, 'FontSize', 10);

% (e) Elevator Actuation
subplot(2,3,5);
plot(t_vec, elev_pid, 'Color', color_pid, 'LineWidth', 1.4); hold on;
plot(t_vec, elev_rl, 'Color', color_rl, 'LineWidth', 1.4);
ylabel('Elevator (norm)');
title('Elevator Actuation');
legend('PID', 'PPO Agent', 'Location', 'best');
grid on; xlim([0 t_end]);
xlabel('Time (s)');
set(gca, 'FontSize', 10);

sgtitle(sprintf('Severity Level %d Turbulence Evaluation', SEVERITY));

% Export at 300 DPI
exportgraphics(fig_combined, sprintf('sev%d_combined_5panel.png', SEVERITY), 'Resolution', 300);
fprintf('Saved: sev%d_combined_5panel.png (300 DPI)\n', SEVERITY);

disp('Done.');
