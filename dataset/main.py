import numpy as np
import matplotlib.pyplot as plt

# Load the clean EEG, EOG, and EMG data
clean_EEG = np.load('dataset/data/EEG_all_epochs.npy') 
EOG = np.load('dataset/data/EOG_all_epochs.npy') 
EMG = np.load('dataset/data/EMG_all_epochs.npy') 

# Define mixing factors
alpha = 0.5  # EOG contribution
beta = 0.3   # EMG contribution

# Determine the smallest number of samples
min_samples = min(clean_EEG.shape[0], EOG.shape[0], EMG.shape[0])

# Truncate all arrays to the smallest shape
clean_EEG = clean_EEG[:min_samples, :]
EOG = EOG[:min_samples, :]
EMG = EMG[:min_samples, :]
# Now they all have the shape (3400, 512)

# Generate noisy EEG
noisy_EEG = clean_EEG + alpha * EOG + beta * EMG

# Basic denoising using moving average 
def moving_average(signal, window_size=5):
    # Apply the moving average to each row of the 2D array
    return np.array([np.convolve(row, np.ones(window_size)/window_size, mode='same') for row in signal])

# Apply moving average
denoised_EEG = moving_average(noisy_EEG)

# Set the number of samples to visualize for clarity
num_samples = 500

# Create a figure with subplots
fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)

# Plot Clean EEG Signal
axes[0].plot(clean_EEG[:num_samples, 0], label='Clean EEG', color='blue')
axes[0].set_title('Clean EEG Signal')
axes[0].set_ylabel('Amplitude')
axes[0].legend()
axes[0].grid()

# Plot EOG Signal
axes[1].plot(EOG[:num_samples, 0], label='EOG Signal', color='orange')
axes[1].set_title('EOG Signal')
axes[1].set_ylabel('Amplitude')
axes[1].legend()
axes[1].grid()

# Plot EMG Signal
axes[2].plot(EMG[:num_samples, 0], label='EMG Signal', color='green')
axes[2].set_title('EMG Signal')
axes[2].set_ylabel('Amplitude')
axes[2].legend()
axes[2].grid()

# Plot Noisy EEG Signal
axes[3].plot(noisy_EEG[:num_samples, 0], label='Noisy EEG', color='red')
axes[3].set_title('Noisy EEG Signal')
axes[3].set_ylabel('Amplitude')
axes[3].legend()
axes[3].grid()

# Plot Denoised EEG Signal
axes[4].plot(denoised_EEG[:num_samples, 0], label='Denoised EEG', color='purple', linestyle='--')
axes[4].set_title('Denoised EEG Signal')
axes[4].set_xlabel('Sample Index')
axes[4].set_ylabel('Amplitude')
axes[4].legend()
axes[4].grid()

# Adjust layout for better spacing
plt.tight_layout()

# Show all plots in one window
plt.show()
