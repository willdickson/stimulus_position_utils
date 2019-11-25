from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import stimulus_position_utils


volt= np.load('data/square/data_downsample.npy')
sample_rate = 100.0  # Guess ... no idea
ind = np.arange(volt.shape[0])
t = ind/sample_rate

ylim = volt.max()*(1.0 + 0.1), volt.min()*(1.0+0.1)

masks = stimulus_position_utils.get_masks_from_square_wave(volt)

# Plot cycles 
# -----------------------------------------------------------------------------
plt.figure(1)
cycle_mask_list = masks['cycles']
print('num cycles: ', len(cycle_mask_list))
for i, cycle_mask in enumerate(cycle_mask_list):
    t_cycle = t[cycle_mask]
    t_cycle = t_cycle - t_cycle[0] 
    plt.plot(t_cycle, volt[cycle_mask], 'b')
plt.ylim(*ylim)
plt.xlabel('t (sec)')
plt.ylabel('voltage')
plt.grid(True)
plt.title('cycles')


# Plot even half cycles 
# -----------------------------------------------------------------------------
plt.figure(2)
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
plt.ylim(*ylim)
plt.title('even half cycles')
plt.xlabel('t (sec)')
plt.ylabel('voltage')

# Plot odd half cycles 
# -----------------------------------------------------------------------------
plt.figure(3)
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
plt.ylim(*ylim)
plt.title('odd half cycles')
plt.xlabel('t (sec)')
plt.ylabel('voltage')


plt.show()


