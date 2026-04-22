% FlightGear Visualisation: RL Agent Flying Cessna 172
% Uses FlightGear's telnet property interface (text-based,
% no binary protocol issues).
%
% INSTRUCTIONS:
% 1. Open Terminal and run:
%    /Applications/FlightGear.app/Contents/MacOS/FlightGear \
%      --fdm=null \
%      --aircraft=c172p \
%      --airport=KSFO \
%      --disable-ai-traffic \
%      --disable-real-weather-fetch \
%      --timeofday=noon \
%      --telnet=socket,bi,10,localhost,5501,tcp
%
% 2. Wait for FlightGear to fully load
% 3. Run this script in MATLAB
% =======================================================
clear all; clc;

dt = 0.02;           % JSBSim timestep
SIM_DURATION = 60;  % seconds
n_steps = floor(SIM_DURATION / dt);
FG_UPDATE_INTERVAL = 3; % send to FG every 3rd step (~17Hz, plenty for viz)

%% Load Agent
disp('Loading Fine-Tuned v2 Agent...');
if exist('curriculum_models/Phase4_finetune/Final_v2_Finetuned.mat', 'file')
    loaded = load('curriculum_models/Phase4_finetune/Final_v2_Finetuned.mat');
    fields = fieldnames(loaded);
    agent = loaded.(fields{1});
    fprintf('Loaded Final_v2_Finetuned.mat\n');
else
    files = dir('curriculum_models/Phase4_finetune/Agent*.mat');
    [~, idx] = max([files.datenum]);
    loaded = load(fullfile(files(idx).folder, files(idx).name));
    fields = fieldnames(loaded);
    agent = loaded.(fields{1});
    fprintf('Loaded: %s\n', files(idx).name);
end
actor = getActor(agent);

%% Setup Environment
env = CessnaMasterEnvv2();
env.CurriculumPhase = 4;
env.TargetAltitude = 4000;
env.TargetSpeed = 100;
env.reset();

fdm = env.fdm;

fdm.set_property_value('ic/h-sl-ft', 4000);
fdm.set_property_value('ic/vc-kts', 100);
fdm.run_ic();

fdm.set_property_value('propulsion/tank[0]/contents-lbs', 150);
fdm.set_property_value('propulsion/tank[1]/contents-lbs', 150);
fdm.set_property_value('fcs/mixture-cmd-norm', 0.85);
fdm.set_property_value('propulsion/magneto_cmd', 3);
fdm.set_property_value('propulsion/starter_cmd', 1);
fdm.set_property_value('propulsion/engine[0]/set-running', 1);

% turbulence
fdm.set_property_value('atmosphere/turb-type', 4);
fdm.set_property_value('atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps', 20);
fdm.set_property_value('atmosphere/turbulence/milspec/severity', 3);
fdm.set_property_value('atmosphere/turbulence/milspec/seed', 12345);

%% Connect to FlightGear via Telnet
disp('Connecting to FlightGear telnet on localhost:5501...');
try
    fg = tcpclient('localhost', 5501, 'ConnectTimeout', 5);
    pause(0.5);
    % Read any welcome message
    if fg.NumBytesAvailable > 0
        read(fg, fg.NumBytesAvailable, 'uint8');
    end
    disp('Connected to FlightGear!');
catch e
    error('Could not connect to FlightGear. Is it running with --telnet=socket,bi,10,localhost,5501,tcp ?');
end

% Helper function to send a property
send_prop = @(name, val) write(fg, uint8(sprintf('set %s %f\r\n', name, val)));

% Calculate warp dynamically to force noon at KSFO
now_utc = datetime('now', 'TimeZone', 'UTC');
noon_ksfo_utc = datetime(year(now_utc), month(now_utc), day(now_utc), 20, 0, 0, 'TimeZone', 'UTC');
warp_secs = round(seconds(noon_ksfo_utc - now_utc));
fprintf('  Applying time warp: %d seconds to reach noon at KSFO\n', warp_secs);
write(fg, uint8(sprintf('set /sim/time/warp %d\r\n', warp_secs)));
pause(0.3);
write(fg, uint8(sprintf('set /sim/freeze/clock true\r\n')));
pause(0.3);
disp('  Clock frozen at noon.');

disp('=== FlightGear Demo Starting ===');
disp('  Press Ctrl+C to stop');
disp(' ');

%% Main Simulation Loop (Real-Time Paced)
tic;
for i = 1:n_steps
    % --- Agent Decision ---
    h = double(fdm.get_property_value('position/h-sl-ft'));
    h_dot = double(fdm.get_property_value('velocities/h-dot-fps'));
    theta = double(fdm.get_property_value('attitude/theta-rad'));
    q = double(fdm.get_property_value('velocities/q-rad_sec'));
    phi = double(fdm.get_property_value('attitude/phi-rad'));
    p = double(fdm.get_property_value('velocities/p-rad_sec'));
    v_kts = double(fdm.get_property_value('velocities/vc-kts'));
    elev_pos = double(fdm.get_property_value('fcs/elevator-cmd-norm'));
    
    err_h = h - env.TargetAltitude;
    err_v = v_kts - env.TargetSpeed;
    obs = [err_h/1000; h_dot/100; theta; q; phi; p; err_v/10; elev_pos];
    
    action = evaluate(actor, {obs});
    act_num = cell2mat(action);
    [~, ~, isDone, ~] = env.step(act_num);
    
    % --- Send to FlightGear (every Nth step) ---
    if mod(i, FG_UPDATE_INTERVAL) == 0
        lat_deg = double(fdm.get_property_value('position/lat-gc-deg'));
        lon_deg = double(fdm.get_property_value('position/long-gc-deg'));
        alt_ft = double(fdm.get_property_value('position/h-sl-ft'));
        phi_deg = rad2deg(double(fdm.get_property_value('attitude/phi-rad')));
        theta_deg = rad2deg(double(fdm.get_property_value('attitude/theta-rad')));
        psi_deg = rad2deg(double(fdm.get_property_value('attitude/psi-rad')));
        
        % Build one batch command string for efficiency
        cmd = sprintf([ ...
            'set /position/latitude-deg %f\r\n' ...
            'set /position/longitude-deg %f\r\n' ...
            'set /position/altitude-ft %f\r\n' ...
            'set /orientation/roll-deg %f\r\n' ...
            'set /orientation/pitch-deg %f\r\n' ...
            'set /orientation/heading-deg %f\r\n' ...
            'set /sim/freeze/clock true\r\n'], ...
            lat_deg, lon_deg, alt_ft, phi_deg, theta_deg, psi_deg);
        
        write(fg, uint8(cmd));
        
        % Drain any responses to prevent buffer buildup
        if fg.NumBytesAvailable > 0
            read(fg, fg.NumBytesAvailable, 'uint8');
        end
    end
    
    % --- Real-Time Pacing ---
    target_time = i * dt;
    elapsed = toc;
    if elapsed < target_time
        pause(target_time - elapsed);
    end
    
    % --- Console Output ---
    if mod(i, 250) == 0
        fprintf('t=%5.1fs | Alt=%6.1f ft | Spd=%5.1f kts | Phi=%+5.1f° | Theta=%+5.1f°\n', ...
            i*dt, h, v_kts, phi_deg, theta_deg);
    end
    
    if isDone
        fprintf('WARNING: Agent terminated at t=%.1fs\n', i*dt);
        break;
    end
end

clear fg;
disp('=== FlightGear Demo Complete ===');
