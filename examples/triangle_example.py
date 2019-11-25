from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import stimulus_position_utils


# Load dataset, truncate for example
#volt = np.load('data/triangle/voltage_downsample.npy')
#volt = np.load('data/triangle/voltage.npy')
#volt = np.load('data/triangle/test_volt.npy')
volt_raw = np.load('data/triangle/pos_rear.npy')
volt = stimulus_position_utils.unwrap(volt_raw)

# Create index and time arrays
sample_rate = 10000.0
ind = np.arange(volt.shape[0])
t = ind/sample_rate

plt.subplot(211)
plt.plot(t,volt_raw)
plt.ylabel('original (V)')
plt.grid(True)

plt.subplot(212)
plt.plot(t,volt)
plt.xlabel('time (s)')
plt.ylabel('unwrapped (V)')
plt.grid(True)
plt.show()

# Parameters for finding cycles
#lower_threshold=0.1          # lower threshold for finding half cycles
#upper_threshold=9.9          # upper threshold for finding half cycles
volt_range = volt.max() - volt.min()
lower_threshold = volt.min() + 0.01*volt_range # lower threshold for finding half cycles
upper_threshold = volt.max() - 0.01*volt_range # upper threshold for finding half cycles
midpoint_window_width=1.0  # width of midpoint voltage window
midpoint_window_min_len=3  # minimum number of points required for midpoint window

masks = stimulus_position_utils.get_masks_from_triangle_wave(
        t,
        volt,
        lower_threshold, 
        upper_threshold, 
        midpoint_window_width, 
        midpoint_window_min_len
        )

#masks = stimulus_position_utils.get_masks(
#        t,
#        volt,
#        lower_threshold = 0.09, 
#        )

## Plot pretrial and trial regions
## -----------------------------------------------------------------------------
plt.figure(1)
pretrial_mask = masks['pretrial']
trial_mask = masks['trial']
plt.plot(t[pretrial_mask],volt[pretrial_mask],'r')
plt.plot(t[trial_mask], volt[trial_mask],'.b')
plt.grid(True)
plt.xlabel('t (sec)')
plt.ylabel('voltage')
plt.title('pretrial and trial')

# Plot cycles 
# -----------------------------------------------------------------------------
plt.figure(2)
cycle_mask_list = masks['cycles']
print('num cycles: ', len(cycle_mask_list))
for i, cycle_mask in enumerate(cycle_mask_list):
    t_cycle = t[cycle_mask]
    t_cycle = t_cycle - t_cycle[0] 
    plt.plot(t_cycle, volt[cycle_mask], 'b')
plt.xlabel('t (sec)')
plt.ylabel('voltage')
plt.grid(True)
plt.title('cycles')

# Create cycle matrix for averaging etc.
# ----------------------------------------------------------------------------
cycle_len_list = []
for i, cycle_mask in enumerate(cycle_mask_list):
    volt_cycle = volt[cycle_mask]
    cycle_len_list.append(volt_cycle.shape[0])

min_cycle_len = min(cycle_len_list)
cycle_matrix = np.zeros((len(cycle_mask_list), min_cycle_len))
for i, cycle_mask in enumerate(cycle_mask_list):
    volt_cycle = volt[cycle_mask][:min_cycle_len]
    cycle_matrix[i,:] = volt_cycle 

# Plot even half cycles 
# -----------------------------------------------------------------------------
plt.figure(3)
even_half_cycle_mask_list = masks['even_half_cycles']
for i, half_cycle_mask in enumerate(even_half_cycle_mask_list):
    if i%2 == 0:
        plot_style = 'b'
    else:
        plot_style = 'g'
    t_half_cycle = t[half_cycle_mask]
    t_half_cycle = t_half_cycle - t_half_cycle[0] # so they all start at t=0
    plt.plot(t_half_cycle, volt[half_cycle_mask], plot_style)
plt.grid(True)
plt.title('even half cycles')
plt.xlabel('t (sec)')
plt.ylabel('voltage')

# Plot odd half cycles 
# -----------------------------------------------------------------------------
plt.figure(4)
odd_half_cycle_mask_list = masks['odd_half_cycles']
for i, half_cycle_mask in enumerate(odd_half_cycle_mask_list):
    if i%2 == 0:
        plot_style = 'b'
    else:
        plot_style = 'g'
    t_half_cycle = t[half_cycle_mask]
    t_half_cycle = t_half_cycle - t_half_cycle[0] # so they all start at t=0
    plt.plot(t_half_cycle, volt[half_cycle_mask], plot_style)
plt.grid(True)
plt.title('odd half cycles')
plt.xlabel('t (sec)')
plt.ylabel('voltage')


plt.show()


