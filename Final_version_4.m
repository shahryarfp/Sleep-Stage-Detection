%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Project Title: EEG Signal Processing for Sleep Stage Detection
% Semi-Supervised Approach used since we don't have labels
%
% Author: Shahryar Namdari
%
% Date: November 7, 2024
%
% Description:
%   This MATLAB script processes an EEG data to identify sleep stages (AWAKE, 
%   REM, NREM). It includes data loading, filtering, wavelet denoising, 
%   feature extraction, dimensionality reduction, and clustering.

%% Load EEG data

clear; clc; close all;

data = load('data.mat');
fs = double(data.dati.fs);  % Sampling frequency
eeg_signal = double(data.dati.eeg);  % EEG signal
eeg_signal = eeg_signal * 1e6; % convert the signal to Microvolts
eeg_signal = eeg_signal - mean(eeg_signal); % Remove DC component (mean subtraction)

%% Bandpass Filter

function filtered = bandpass_filter(low_cutoff, high_cutoff, signal, fs)
    order = 4;
    [b, a] = butter(order, [low_cutoff, high_cutoff] / (fs / 2), 'bandpass');
    filtered = filtfilt(b, a, signal);
end
filtered_eeg = bandpass_filter(1, 40, eeg_signal, fs);

%% Wavelet Denoising

waveletName = 'db8';
level = 7;
[c, l] = wavedec(filtered_eeg, level, waveletName);
sigma = median(abs(c)) / 0.6745;
threshold = sigma * sqrt(2 * log(length(filtered_eeg)));
c_denoised = wthresh(c, 's', threshold);
denoised_eeg = waverec(c_denoised, l, waveletName);

%% Reapply Bandpass Filter to Remove Residual Noise

denoised_eeg = bandpass_filter(1, 40, denoised_eeg, fs);

%% Plot the original and cleaned EEG signals

t = (0:length(eeg_signal)-1) / fs;
figure;
subplot(2, 1, 1);
plot(t, eeg_signal);
title('Original EEG Signal');
xlabel('Time (s)');
ylabel('Amplitude (μV)');

subplot(2, 1, 2);
plot(t, denoised_eeg);
title('Cleaned EEG Signal after Artifact Removal');
xlabel('Time (s)');
ylabel('Amplitude (μV)');

sig = denoised_eeg;

%% plot FFT and PSD

fft_sig = fft(sig);
N = length(sig);
figure;
subplot(2,2,1);
plot((0:N-1)*(fs/N), abs(fft_sig)/N);
title('FFT of the Signal');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([0 fs/2]);
grid on;
grid minor;

subplot(2,2,3);
overlap_size = length(sig) * 0.75;
[psd, f] = pwelch(sig, hamming(length(sig)), overlap_size, [], fs);
plot(f, psd);
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density');
title("PSD using Welch Method");
grid on;
grid minor;

%% Bandstop (Notch) Filter

[b1, a1] = butter(2, [0.999 1.001] / (fs / 2), 'stop'); % Notch filter for 1 Hz
[b2, a2] = butter(2, [1.999 2.001] / (fs / 2), 'stop'); % Notch filter for 2 Hz

sig = filtfilt(b1, a1, sig);
sig = filtfilt(b2, a2, sig);

%% plot FFT and PSD after notch filter

fft_sig = fft(sig);
N = length(sig);
subplot(2,2,2);
plot((0:N-1)*(fs/N), abs(fft_sig)/N);
title('FFT of the Signal after Notch Filter');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([0 fs/2]);
grid on;
grid minor;

subplot(2,2,4);
overlap_size = round(length(sig) * 0.25);
[psd, f] = pwelch(sig, hamming(length(sig)), overlap_size, [], fs);
plot(f, psd);
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density');
title("PSD using Welch Method after Notch Filter");
grid on;
grid minor;

%% plot rythms

rhythms = {
    'Delta', [1, 4];
    'Theta', [5, 7];
    'Alpha', [8, 12];
    'Sigma', [12, 14];
    'Beta', [14, 30];
    'Gamma', [30, 40];
};

figure;
subplot(length(rhythms)+1,1,1);
plot(sig);
title("Original EEG Signal Before Filtering");
xlabel('Samples');
ylabel('Amplitude (μV)');

for i = 1:length(rhythms)
    % Bandpass filter design
    order = 3;
    fl = rhythms{i, 2}(1);
    fh = rhythms{i, 2}(2);
    wn = [fl fh] / (fs / 2);
    [b, a] = butter(order, wn, 'bandpass');
    
    % Filter the signal and plot
    filtered_sig = filtfilt(b, a, sig);
    subplot(length(rhythms)+1, 1, i+1);
    plot(filtered_sig);
    title([rhythms{i, 1}, ' Wave']);
    xlabel('Samples');
    ylabel('Amplitude (μV)');
end

%% feature extraction: Time and Frequency Domain

function features = fextract_time(sig)
    sk = skewness(sig);
    zcr = zerocrossrate(sig);
    features = [sk, zcr];
end

function features = fextract_freq(psd_band, total_power, psd, f, bound)
    sk = skewness(psd_band);
    kr = kurtosis(psd_band);
    [~, idx] = max(psd_band);
    peak_freq = f(idx + find(f >= bound(1), 1) - 1);
    BP = bandpower(psd, f, bound, 'psd');
    normalized_power_ratio = BP / total_power;
    features = [normalized_power_ratio, sk, kr, peak_freq, BP];
end

window_length_in_sec = 30;
window_size = window_length_in_sec * fs;
overlap_size = 0.5 * window_size; % 50% overlap
eeg_segments = buffer(sig, window_size, overlap_size, 'nodelay');

num_windows = size(eeg_segments, 2);
num_time_features = 2;
num_freq_features_per_rhythm = 5;
num_rhythms = 5;
total_features = num_rhythms * num_freq_features_per_rhythm + num_time_features;

feature_matrix = zeros(num_windows, total_features);

for i = 1:num_windows
    segment = eeg_segments(:, i);

    % Time-domain
    time_features = fextract_time(segment);
    
    % Frequency-domain
    % PSD Feature Extraction using Welch's method
    overlap_size = length(segment) * 0.75;
    [psd, f] = pwelch(segment, hamming(length(segment)), overlap_size, [], fs);
    delta_idx = find(f >= 1 & f <= 4);
    theta_idx = find(f >= 5 & f <= 7);
    alpha_idx = find(f >= 8 & f <= 12);
    sigma_idx = find(f >= 12 & f <= 14);
    beta_idx = find(f >= 14 & f <= 30);
    
    total_power = sum(psd);

    % Extract features for each frequency band using PSD
    delta_features = fextract_freq(psd(delta_idx), total_power, psd, f, [1, 4]);
    theta_features = fextract_freq(psd(theta_idx), total_power, psd, f, [5, 7]);
    alpha_features = fextract_freq(psd(alpha_idx), total_power, psd, f, [8, 12]);
    sigma_features = fextract_freq(psd(sigma_idx), total_power, psd, f, [12, 14]);
    beta_features = fextract_freq(psd(beta_idx), total_power, psd, f, [14, 30]);
    
    feature_vector = [delta_features, theta_features, alpha_features, sigma_features, beta_features, time_features];
    feature_matrix(i, :) = feature_vector;
end

% Calculate and add relative bandpower ratios to the feature matrix
% Extract each Band-Power from the feature matrix
delta_power = feature_matrix(:, 5);
theta_power = feature_matrix(:, 10);
alpha_power = feature_matrix(:, 15);
beta_power = feature_matrix(:, 25);

% Calculate relative bandpower ratios
delta_beta_ratio = delta_power ./ beta_power;
theta_beta_ratio = theta_power ./ beta_power;
alpha_beta_ratio = alpha_power ./ beta_power;

% Combine the ratios into a new feature matrix
relative_ratios = [delta_beta_ratio, theta_beta_ratio, alpha_beta_ratio];
feature_matrix = [feature_matrix, relative_ratios];

%% Determine thresholds for pre-labeling

% Plot histograms for each ratio
figure;

subplot(3, 1, 1);
histogram(delta_beta_ratio, 30);
title('Distribution of Delta/Beta Ratio');
xlabel('Delta/Beta Ratio');
ylabel('Frequency');

subplot(3, 1, 2);
histogram(theta_beta_ratio, 30);
title('Distribution of Theta/Beta Ratio');
xlabel('Theta/Beta Ratio');
ylabel('Frequency');

subplot(3, 1, 3);
histogram(alpha_beta_ratio, 30);
title('Distribution of Alpha/Beta Ratio');
xlabel('Alpha/Beta Ratio');
ylabel('Frequency');

% Calculate mean, median, and percentiles for each ratio
fprintf('Delta/Beta Ratio:\n');
fprintf('Mean: %.2f, Median: %.2f, 25th percentile: %.2f, 75th percentile: %.2f\n', ...
    mean(delta_beta_ratio), median(delta_beta_ratio), prctile(delta_beta_ratio, 25), prctile(delta_beta_ratio, 75));

fprintf('\nTheta/Beta Ratio:\n');
fprintf('Mean: %.2f, Median: %.2f, 25th percentile: %.2f, 75th percentile: %.2f\n', ...
    mean(theta_beta_ratio), median(theta_beta_ratio), prctile(theta_beta_ratio, 25), prctile(theta_beta_ratio, 75));

fprintf('\nAlpha/Beta Ratio:\n');
fprintf('Mean: %.2f, Median: %.2f, 25th percentile: %.2f, 75th percentile: %.2f\n', ...
    mean(alpha_beta_ratio), median(alpha_beta_ratio), prctile(alpha_beta_ratio, 25), prctile(alpha_beta_ratio, 75));

%% Standardize (Log Transformation Followed by Global Scaling)

% Remove band power features since their raw values are useless!
feature_matrix(:, [5, 10, 15, 20, 25]) = [];

% Shift the data to make all values positive
shift_constant = abs(min(feature_matrix(:))) + 1e-6;
shifted_features = feature_matrix + shift_constant;
log_features = log(shifted_features + 1e-6);

% Global min-max normalization on log-transformed features
global_min = min(log_features(:));
global_max = max(log_features(:));
standardized_features = (log_features - global_min) / (global_max - global_min);

%% Removing highly correlated features

corr_matrix = corr(standardized_features);
high_corr_threshold = 0.98;
high_corr = abs(corr_matrix) > high_corr_threshold;

% Keep only one feature from each correlated pair
[rows, cols] = find(tril(high_corr, -1));
to_remove = unique(cols);
standardized_features(:, to_remove) = [];
fprintf('\nRemoved high-corrolated features:');
disp(to_remove');

%% Dimensionality Reduction (PCA)

[coeff, score, latent, ~, explained] = pca(standardized_features);
cumulative_variance = cumsum(explained);
% Get number of components for 98% variance
num_components = find(cumulative_variance >= 98, 1);

% Extract the principal component coefficients
coeff_selected = coeff(:, 1:num_components);
% Find the most contributing original features for each principal component
[~, most_contributing_features] = max(abs(coeff_selected), [], 1);
% Display the indices of the original features that contribute most to each component
fprintf('Most contributing original feature for each principal component:');
disp(most_contributing_features);

reduced_feature_matrix = score(:, 1:num_components);

% Explained variance
figure;
plot(cumulative_variance, 'o-');
xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance (%)');
title('PCA Explained Variance');
grid on;

%% Pre-Label Data Segments Based on Knowledge-Based Rules
% Apply empirically determined threshold rules for pre-labeling sleep stages
% 1 = AWAKE, 2 = REM, 3 = NREM

% Initialize the pre_labels vector
pre_labels = zeros(size(relative_ratios, 1), 1);

% Apply pre-labeling based on threshold criteria
for i = 1:length(pre_labels)
    if delta_beta_ratio(i) > 700 && alpha_beta_ratio(i) < 3
        pre_labels(i) = 3; % NREM

    elseif theta_beta_ratio(i) > 20 && theta_beta_ratio(i) < 60 && delta_beta_ratio(i) < 300 && alpha_beta_ratio(i) < 5
        pre_labels(i) = 2; % REM
        
    elseif alpha_beta_ratio(i) > 6 && delta_beta_ratio(i) < 20
        pre_labels(i) = 1; % AWAKE
    end
end

% Display the number of samples in each pre-labeled category
disp('Pre-label counts:');
disp(['AWAKE: ', num2str(sum(pre_labels == 1))]);
disp(['REM: ', num2str(sum(pre_labels == 2))]);
disp(['NREM: ', num2str(sum(pre_labels == 3))]);

%% Unsupervised Clustering
% Apply Gaussian Mixture Model (GMM) for Unsupervised Clustering of Sleep Stages
% for AWAKE, REM, and NREM

final_feature_matrix = reduced_feature_matrix;
numClusters = 3;

% Initialize GMM with pre-labeled group means and covariance
initialMeans = zeros(numClusters, size(final_feature_matrix, 2));
initialCovariance = zeros(size(final_feature_matrix, 2));  % Shared covariance matrix

for k = 1:numClusters
    clusterData = final_feature_matrix(pre_labels == k, :);
    initialMeans(k, :) = mean(clusterData, 1);  % Mean of each pre-labeled cluster
    initialCovariance = initialCovariance + cov(clusterData);  % Accumulate covariances
end

% Average covariance matrix for a shared initial covariance
initialCovariance = initialCovariance / numClusters;

% Set options for GMM fitting
options = statset('MaxIter', 1000, 'Display', 'final');

% Fit GMM with initial parameters and shared covariance
gmmModel = fitgmdist(final_feature_matrix, numClusters, ...
    'Start', struct('mu', initialMeans, 'Sigma', initialCovariance), ...
    'Options', options, 'SharedCovariance', true, 'Regularize', 1e-5);

% Cluster data based on the GMM model
idx = cluster(gmmModel, final_feature_matrix);

% Display counts for each cluster
disp('Final GMM cluster counts:');
disp(['Cluster 1 (AWAKE): ', num2str(sum(idx == 1))]);
disp(['Cluster 2 (REM): ', num2str(sum(idx == 2))]);
disp(['Cluster 3 (NREM): ', num2str(sum(idx == 3))]);

%% Evaluation

% % Using silhouette score
% figure;
% [silh, h] = silhouette(final_feature_matrix, idx);
% avg_silh = mean(silh);
% title(['Silhouette Plot for K-means Clustering (Average Width = ', num2str(avg_silh), ')']);

% Davies-Bouldin Index (DBI)
dbi_score = evalclusters(final_feature_matrix, idx, 'DaviesBouldin');
fprintf('Davies-Bouldin Index: %.3f\n', dbi_score.CriterionValues);

%% Segment the Original EEG Signal Based on Clustering Results

clusters = {1, 2, 3};
values = {'AWAKE', 'REM', 'NREM'};
main_dic = containers.Map(clusters, values);
reverse_main_dic = containers.Map(values, clusters);

segment_length = size(eeg_segments, 1);
labeled_signal = zeros(size(sig));

for i = 1:num_windows
    start_idx = (i - 1) * segment_length + 1;
    end_idx = min(i * segment_length, length(sig));
    labeled_signal(start_idx:end_idx) = idx(i);
end

time_axis = (1:length(sig)) / fs;

% Plot
figure;
subplot(2, 1, 1);
plot(time_axis, sig, 'k');
xlabel('Time (seconds)');
ylabel('EEG Signal');
title('Original EEG Signal');
legend('Original Signal', 'FontSize', 12);
grid on;

subplot(2, 1, 2);
plot(time_axis, sig, 'k');
hold on;

% Highlight different sleep stages

% Define colors for background shading
awake_color = [0.2, 0.6, 0.2];
rem_color = [0.9, 0.3, 0.3];
nrem_color = [0.3, 0.5, 0.8];

% Handles for the legend
h_awake = fill(nan, nan, awake_color, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
h_rem = fill(nan, nan, rem_color, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
h_nrem = fill(nan, nan, nrem_color, 'FaceAlpha', 0.2, 'EdgeColor', 'none');

dic0 = containers.Map({'AWAKE','REM','NREM'}, {awake_color, rem_color, nrem_color});
dic1 = containers.Map({'AWAKE','REM','NREM'}, {h_awake, h_rem, h_nrem});

for i = 1:num_windows
    start_idx = (i - 1) * segment_length + 1;
    end_idx = min(i * segment_length, length(sig));
    % Ensure indices stay within bounds
    if start_idx > length(sig) || end_idx > length(sig)
        break;
    end
    cluster_label = idx(i);
    % Plot the segment with a different color for each cluster
    if cluster_label == 1
        fill([time_axis(start_idx), time_axis(end_idx), time_axis(end_idx), time_axis(start_idx)], ...
             [min(sig), min(sig), max(sig), max(sig)], dic0(main_dic(1)), 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    elseif cluster_label == 2
        fill([time_axis(start_idx), time_axis(end_idx), time_axis(end_idx), time_axis(start_idx)], ...
             [min(sig), min(sig), max(sig), max(sig)], dic0(main_dic(2)), 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    elseif cluster_label == 3
        fill([time_axis(start_idx), time_axis(end_idx), time_axis(end_idx), time_axis(start_idx)], ...
             [min(sig), min(sig), max(sig), max(sig)], dic0(main_dic(3)), 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    end
end
xlabel('Time (seconds)');
ylabel('EEG Signal');
title('Segmented EEG Signal by Sleep Stage');
legend([dic1(main_dic(1)), dic1(main_dic(2)), dic1(main_dic(3))], {main_dic(1), main_dic(2), main_dic(3)}, 'FontSize', 12);
grid on;
hold off;
%% Analyze Each Cluster Separately Based on Features

% Colors for Delta, Theta, and Beta Power
colors = {[0.3, 0.5, 0.8], [0.9, 0.5, 0.2], [0.8, 0.8, 0.2]};
num_clusters = 3;

figure;

% Delta/Beta Ratio
subplot(3, 1, 1);
hold on;
for c = 1:num_clusters
    histogram(relative_ratios(idx == c, 1), 'FaceColor', colors{c}, 'FaceAlpha', 0.6, 'DisplayName', sprintf('Cluster %d', c));
    mean_delta_beta = mean(relative_ratios(idx == c, 1));
    xline(mean_delta_beta, '--', 'Color', colors{c}, 'LineWidth', 1.5, 'DisplayName', sprintf('Mean Cluster %d', c));
end
title('Delta/Beta Ratio Distribution Across Clusters');
xlabel('Delta/Beta Ratio');
ylabel('Frequency');
legend;
hold off;

% Theta/Beta Ratio
subplot(3, 1, 2);
hold on;
for c = 1:num_clusters
    histogram(relative_ratios(idx == c, 2), 'FaceColor', colors{c}, 'FaceAlpha', 0.6, 'DisplayName', sprintf('Cluster %d', c));
    mean_theta_beta = mean(relative_ratios(idx == c, 2));
    xline(mean_theta_beta, '--', 'Color', colors{c}, 'LineWidth', 1.5, 'DisplayName', sprintf('Mean Cluster %d', c));
end
title('Theta/Beta Ratio Distribution Across Clusters');
xlabel('Theta/Beta Ratio');
ylabel('Frequency');
legend;
hold off;

% Alpha/Beta Ratio
subplot(3, 1, 3);
hold on;
for c = 1:num_clusters
    histogram(relative_ratios(idx == c, 3), 'FaceColor', colors{c}, 'FaceAlpha', 0.6, 'DisplayName', sprintf('Cluster %d', c));
    mean_alpha_beta = mean(relative_ratios(idx == c, 3));
    xline(mean_alpha_beta, '--', 'Color', colors{c}, 'LineWidth', 1.5, 'DisplayName', sprintf('Mean Cluster %d', c));
end
title('Alpha/Beta Ratio Distribution Across Clusters');
xlabel('Alpha/Beta Ratio');
ylabel('Frequency');
legend;
hold off;

%% Hypnogram

time_axis = (1:length(sig)) / fs / 3600;

figure;
hold on;

% Plot background for each sleep stage
fill([time_axis(1), time_axis(end), time_axis(end), time_axis(1)], ...
     [3.5, 3.5, 2.5, 2.5], nrem_color, 'EdgeColor', 'none', 'FaceAlpha', 0.3);
fill([time_axis(1), time_axis(end), time_axis(end), time_axis(1)], ...
     [2.5, 2.5, 1.5, 1.5], rem_color, 'EdgeColor', 'none', 'FaceAlpha', 0.3);
fill([time_axis(1), time_axis(end), time_axis(end), time_axis(1)], ...
     [1.5, 1.5, 0.5, 0.5], awake_color, 'EdgeColor', 'none', 'FaceAlpha', 0.3);

hypnogram_values = zeros(length(sig), 1);

% Plot each sleep stage with a line indicating transitions
dic2 = containers.Map({'AWAKE', 'REM', 'NREM'}, {1,2,3});
for i = 1:num_windows
    start_idx = (i - 1) * segment_length + 1;
    end_idx = min(i * segment_length, length(sig));
    cluster_label = idx(i);
    if cluster_label == 1
        % REM stage in the middle
        hypnogram_values(start_idx:end_idx) = dic2(main_dic(1));
    elseif cluster_label == 2
        % Awake stage at the top
        hypnogram_values(start_idx:end_idx) = dic2(main_dic(2));
    elseif cluster_label == 3
        % NREM stage at the bottom
        hypnogram_values(start_idx:end_idx) = dic2(main_dic(3));
    end
end

% Connecting lines for sleep stages
plot(time_axis, hypnogram_values, 'k-', 'LineWidth', 0.5);

yticks([1 2 3]); % Use increasing order
yticklabels({'AWAKE', 'ٔٔREM', 'NREM'});
xlabel('Time (hours)');
ylabel('Sleep Stage');
title('Sleep Stage Hypnogram');
grid on;
grid minor;
xlim([time_axis(1), time_axis(end)]);
hold off;
set(gca, 'YDir', 'reverse');

%% The End

disp('EEG analysis completed successfully.')