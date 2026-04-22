% scenElevatorIcing.m
% Elevator Icing + Vertical Gust Stress Test
%
% Simulates progressive elevator control surface icing degradation
% followed by a moderate vertical gust that would normally be manageable
% but becomes dangerous with compromised control authority.
%
% This replicates a well-documented accident chain in GA:
%   - Ice accretion reduces control surface effectiveness
%   - Aircraft encounters turbulence/gusts that would normally be benign
%   - Reduced authority prevents adequate recovery
%   (ref: FAA Safety Alert SA-014, NTSB accident database)
%
% Scenario Timeline:
%   Phase 1 (0-20s):    Calm baseline — full control authority
%   Phase 2 (20-40s):   Gradual icing — elevator degrades 100% -> 40%
%   Phase 3 (40-60s):   Iced cruise — degraded authority, no disturbance
%   Phase 4 (60-75s):   Moderate downdraft at 25fps with degraded controls
%   Phase 5 (75-120s):  Calm recovery with degraded controls

clear all; clc; close all;

%% SCENARIO PARAMETERS
ICING_START     = 20;    % When elevator begins degrading (s)
ICING_END       = 40;    % When degradation reaches minimum (s)
MIN_AUTHORITY   = 0.40;  % Final elevator effectiveness (40%)
GUST_START      = 60;    % Downdraft onset (s)
GUST_END        = 75;    % Downdraft end (s)
GUST_MAG_FPS    = 25;    % Downdraft magnitude (fps) — moderate, not extreme

% Safety envelopes
STALL_SPEED_KTS = 55;
MIN_SAFE_ALT    = 3200;
MAX_PITCH_DEG   = 30;

t_end = 300; dt = 0.02;
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
    
    fdm.set_property_value('atmosphere/turb-type', 0);
    fdm.set_property_value('atmosphere/wind-north-fps', 0);
    fdm.set_property_value('atmosphere/wind-east-fps', 0);
    fdm.set_property_value('atmosphere/wind-down-fps', 0);
    
    fdm.set_property_value('fcs/throttle-cmd-norm', 0.80);
    for k = 1:25, fdm.run(); end
end

%% Compute elevator authority at time t
function auth = get_authority(t, icing_start, icing_end, min_auth)
    if t < icing_start
        auth = 1.0;
    elseif t < icing_end
        frac = (t - icing_start) / (icing_end - icing_start);
        auth = 1.0 - frac * (1.0 - min_auth);  % linear ramp down
    else
        auth = min_auth;
    end
end

%% ==================== PID CONTROLLER ====================
disp('Running PID controller...');
fdm_pid = init_jsbsim_calm();

alt_pid   = zeros(n_steps, 1);
spd_pid   = zeros(n_steps, 1);
elev_pid  = zeros(n_steps, 1);  % What PID computes
elev_applied_pid = zeros(n_steps, 1);  % What actually reaches the surface
theta_pid = zeros(n_steps, 1);
thr_pid   = zeros(n_steps, 1);
auth_rec  = zeros(n_steps, 1);
wind_rec  = zeros(n_steps, 1);

int_h = 0; int_phi = 0;
pid_crashed = false; pid_crash_time = NaN; pid_crash_reason = '';

for i = 1:n_steps
    t = t_vec(i);
    
    % Elevator authority degradation
    authority = get_authority(t, ICING_START, ICING_END, MIN_AUTHORITY);
    auth_rec(i) = authority;
    
    % Vertical gust
    if t >= GUST_START && t < GUST_END
        fdm_pid.set_property_value('atmosphere/wind-down-fps', GUST_MAG_FPS);
        wind_rec(i) = GUST_MAG_FPS;
    else
        fdm_pid.set_property_value('atmosphere/wind-down-fps', 0);
        wind_rec(i) = 0;
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
    
    % Crash detection
    if ~pid_crashed
        if h < MIN_SAFE_ALT
            pid_crashed = true; pid_crash_time = t;
            pid_crash_reason = sprintf('ALTITUDE LOSS: %.0f ft', h);
        elseif v < STALL_SPEED_KTS
            pid_crashed = true; pid_crash_time = t;
            pid_crash_reason = sprintf('STALL: %.1f kts', v);
        elseif abs(rad2deg(theta)) > MAX_PITCH_DEG
            pid_crashed = true; pid_crash_time = t;
            pid_crash_reason = sprintf('EXTREME ATTITUDE: %.1f deg', rad2deg(theta));
        end
    end
    
    % Throttle
    err_v = 100 - v;
    thr_cmd = max(min(0.65 + (0.02 * err_v), 1.0), 0.0);
    fdm_pid.set_property_value('fcs/throttle-cmd-norm', thr_cmd);
    
    % Elevator (PID computes, but icing reduces what reaches the surface)
    err_h = 4000 - h;
    int_h = max(min(int_h + (err_h * dt), 20), -20);
    theta_cmd = max(min((kp_h*err_h) + (ki_h*int_h) - (kd_h*h_dot), 0.087), -0.15);
    elev_cmd = max(min(POLARITY * (kp_theta*(theta_cmd-theta) - kd_q*q), 1), -1);
    elev_actual = elev_cmd * authority;  % ICING DEGRADES THE COMMAND
    fdm_pid.set_property_value('fcs/elevator-cmd-norm', elev_actual);
    
    % Lateral
    err_psi = 0 - psi;
    phi_cmd = max(min(kp_psi * err_psi, 0.35), -0.35);
    err_phi = phi_cmd - phi;
    int_phi = max(min(int_phi + (err_phi * dt), 0.5), -0.5);
    ail = max(min((kp_phi*err_phi) + (ki_phi*int_phi) - (kd_p*p), 1), -1);
    fdm_pid.set_property_value('fcs/aileron-cmd-norm', ail);
    
    fdm_pid.run();
    
    alt_pid(i) = h; spd_pid(i) = v;
    elev_pid(i) = elev_cmd; elev_applied_pid(i) = elev_actual;
    theta_pid(i) = rad2deg(theta); thr_pid(i) = thr_cmd;
end
disp('  PID complete.');

%% ==================== RL AGENT ====================
disp('Running RL Agent...');
fdm_rl = init_jsbsim_calm();

alt_rl   = zeros(n_steps, 1);
spd_rl   = zeros(n_steps, 1);
elev_rl  = zeros(n_steps, 1);  % What RL commands
elev_applied_rl = zeros(n_steps, 1);  % What reaches the surface
theta_rl = zeros(n_steps, 1);
thr_rl   = zeros(n_steps, 1);

alpha_filter = 0.05; prev_elev = 0;
rl_crashed = false; rl_crash_time = NaN; rl_crash_reason = '';

for i = 1:n_steps
    t = t_vec(i);
    
    % IDENTICAL icing and gust schedule
    authority = get_authority(t, ICING_START, ICING_END, MIN_AUTHORITY);
    
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
    
    % Crash detection
    if ~rl_crashed
        if h < MIN_SAFE_ALT
            rl_crashed = true; rl_crash_time = t;
            rl_crash_reason = sprintf('ALTITUDE LOSS: %.0f ft', h);
        elseif v < STALL_SPEED_KTS
            rl_crashed = true; rl_crash_time = t;
            rl_crash_reason = sprintf('STALL: %.1f kts', v);
        elseif abs(rad2deg(th)) > MAX_PITCH_DEG
            rl_crashed = true; rl_crash_time = t;
            rl_crash_reason = sprintf('EXTREME ATTITUDE: %.1f deg', rad2deg(th));
        end
    end
    
    err_h_c = h - 4000;
    err_v_c = v - 100;
    obs = [err_h_c/1000; h_dot/100; th; q; phi; p; err_v_c/10; elev_pos];
    
    action = evaluate(actor, {obs});
    act = cell2mat(action);
    
    elev_raw = max(min(act(1), 1), -1);
    elev_cmd = alpha_filter * elev_raw + (1 - alpha_filter) * prev_elev;
    prev_elev = elev_cmd;
    elev_actual = elev_cmd * authority;  % SAME ICING DEGRADATION
    
    ail = max(min(act(2), 1), -1);
    thr_raw = max(min(act(3), 1), -1);
    thr = (thr_raw + 1) / 2;
    
    fdm_rl.set_property_value('fcs/elevator-cmd-norm', elev_actual);
    fdm_rl.set_property_value('fcs/aileron-cmd-norm', ail);
    fdm_rl.set_property_value('fcs/throttle-cmd-norm', thr);
    fdm_rl.run();
    
    alt_rl(i) = h; spd_rl(i) = v;
    elev_rl(i) = elev_cmd; elev_applied_rl(i) = elev_actual;
    theta_rl(i) = rad2deg(th); thr_rl(i) = thr;
end
disp('  RL complete.');

%% ==================== CRASH REPORT ====================
disp(' ');
disp('========== CRASH ANALYSIS ==========');
if pid_crashed
    fprintf('PID: *** CRASHED *** at t = %.1f s — %s\n', pid_crash_time, pid_crash_reason);
else
    fprintf('PID: SURVIVED (min alt: %.0f ft, min speed: %.1f kts, max pitch: %.1f deg)\n', ...
        min(alt_pid), min(spd_pid), max(abs(theta_pid)));
end
if rl_crashed
    fprintf('RL:  *** CRASHED *** at t = %.1f s — %s\n', rl_crash_time, rl_crash_reason);
else
    fprintf('RL:  SURVIVED (min alt: %.0f ft, min speed: %.1f kts, max pitch: %.1f deg)\n', ...
        min(alt_rl), min(spd_rl), max(abs(theta_rl)));
end
fprintf('\nElevator authority at gust onset: %.0f%%\n', MIN_AUTHORITY * 100);
fprintf('Peak altitude deviation PID: %.0f ft\n', max(abs(alt_pid - 4000)));
fprintf('Peak altitude deviation RL:  %.0f ft\n', max(abs(alt_rl - 4000)));
disp('=====================================');

%% ==================== PLOTTING ====================
color_pid = [0.33, 0.49, 0.74];
color_rl  = [0.42, 0.71, 0.43];

fig = figure('Position', [50, 50, 1200, 900], 'Color', 'w');

% Phase shading helper
gust_x = [GUST_START GUST_END GUST_END GUST_START];
ice_x  = [ICING_START ICING_END ICING_END ICING_START];

% 1. ALTITUDE
subplot(3,2,1);
yl = [min([min(alt_pid), min(alt_rl)]) - 50, max([max(alt_pid), max(alt_rl)]) + 50];
fill(ice_x, [yl(1) yl(1) yl(2) yl(2)], [0.85 0.85 0.95], 'EdgeColor', 'none', 'FaceAlpha', 0.3); hold on;
fill(gust_x, [yl(1) yl(1) yl(2) yl(2)], [0.95 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(t_vec, alt_pid, 'Color', color_pid, 'LineWidth', 1.5);
plot(t_vec, alt_rl, 'Color', color_rl, 'LineWidth', 1.5);
yline(4000, 'k--', 'LineWidth', 0.8);
yline(MIN_SAFE_ALT, 'r--', 'LineWidth', 1, 'Label', 'Safety Floor');
ylim(yl);
title('Altitude Response'); ylabel('Altitude (ft)'); xlabel('Time (s)'); grid on;
legend('Icing Phase', 'Downdraft', 'PID', 'RL', 'Location', 'best');

% 2. AIRSPEED
subplot(3,2,2);
yl2 = [min([min(spd_pid), min(spd_rl)]) - 5, max([max(spd_pid), max(spd_rl)]) + 5];
fill(ice_x, [yl2(1) yl2(1) yl2(2) yl2(2)], [0.85 0.85 0.95], 'EdgeColor', 'none', 'FaceAlpha', 0.3); hold on;
fill(gust_x, [yl2(1) yl2(1) yl2(2) yl2(2)], [0.95 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(t_vec, spd_pid, 'Color', color_pid, 'LineWidth', 1.5);
plot(t_vec, spd_rl, 'Color', color_rl, 'LineWidth', 1.5);
yline(100, 'k--', 'LineWidth', 0.8);
yline(STALL_SPEED_KTS, 'r--', 'LineWidth', 1, 'Label', 'Stall');
title('Airspeed Response'); ylabel('Airspeed (kts)'); xlabel('Time (s)'); grid on;

% 3. PITCH
subplot(3,2,3);
yl3 = [min([min(theta_pid), min(theta_rl)]) - 2, max([max(theta_pid), max(theta_rl)]) + 2];
fill(ice_x, [yl3(1) yl3(1) yl3(2) yl3(2)], [0.85 0.85 0.95], 'EdgeColor', 'none', 'FaceAlpha', 0.3); hold on;
fill(gust_x, [yl3(1) yl3(1) yl3(2) yl3(2)], [0.95 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(t_vec, theta_pid, 'Color', color_pid, 'LineWidth', 1.5);
plot(t_vec, theta_rl, 'Color', color_rl, 'LineWidth', 1.5);
yline(0, 'k--', 'LineWidth', 0.5);
title('Pitch Attitude'); ylabel('Pitch (\circ)'); xlabel('Time (s)'); grid on;

% 4. ELEVATOR — Commanded vs Applied
subplot(3,2,4);
fill(ice_x, [-1.2 -1.2 1.2 1.2], [0.85 0.85 0.95], 'EdgeColor', 'none', 'FaceAlpha', 0.3); hold on;
fill(gust_x, [-1.2 -1.2 1.2 1.2], [0.95 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(t_vec, elev_pid, '--', 'Color', color_pid, 'LineWidth', 0.8);
plot(t_vec, elev_applied_pid, '-', 'Color', color_pid, 'LineWidth', 1.5);
plot(t_vec, elev_rl, '--', 'Color', color_rl, 'LineWidth', 0.8);
plot(t_vec, elev_applied_rl, '-', 'Color', color_rl, 'LineWidth', 1.5);
title('Elevator (dashed=commanded, solid=applied)');
ylabel('Elevator (norm)'); xlabel('Time (s)'); grid on;

% 5. THROTTLE
subplot(3,2,5);
fill(ice_x, [-0.1 -0.1 1.1 1.1], [0.85 0.85 0.95], 'EdgeColor', 'none', 'FaceAlpha', 0.3); hold on;
fill(gust_x, [-0.1 -0.1 1.1 1.1], [0.95 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(t_vec, thr_pid, 'Color', color_pid, 'LineWidth', 1.2);
plot(t_vec, thr_rl, 'Color', color_rl, 'LineWidth', 1.2);
title('Throttle Command'); ylabel('Throttle (norm)'); xlabel('Time (s)'); grid on;

% 6. ELEVATOR AUTHORITY
subplot(3,2,6);
fill(ice_x, [-0.1 -0.1 1.1 1.1], [0.85 0.85 0.95], 'EdgeColor', 'none', 'FaceAlpha', 0.3); hold on;
fill(gust_x, [-0.1 -0.1 1.1 1.1], [0.95 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(t_vec, auth_rec * 100, 'Color', [0.7 0.2 0.2], 'LineWidth', 2);
ylim([0 110]);
title('Elevator Authority (Icing Degradation)');
ylabel('Authority (%)'); xlabel('Time (s)'); grid on;

sgtitle(sprintf('Elevator Icing + Downdraft Test (%.0f%% authority, %.0f fps gust)', MIN_AUTHORITY*100, GUST_MAG_FPS));

exportgraphics(fig, 'scenElevatorIcing.png', 'Resolution', 300);
disp('Saved: scenElevatorIcing.png (300 DPI)');
disp('Done.');
