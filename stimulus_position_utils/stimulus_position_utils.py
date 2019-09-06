from __future__ import print_function
import numpy as np


def get_masks(
        t,                            # array of time points
        volt,                         # array of stimulus voltage values
        lower_threshold=0.1,          # lower threshold for finding half cycles
        upper_threshold=9.9,          # upper threshold for finding half cycles
        midpoint_window_width=1.0,    # width of midpoint voltage window
        midpoint_window_min_len=3     # minimum number of points required for midpoint window
        ): 
    
    """" 
    Get masks for pretrial, trial regions and lists of masks for full and half
    cycles based on the stimulus voltage.  

    Expects a triangle wave type stimulus voltage w/ some hinkiness at the
    extrema.  The half cycles are identified by finding the regions where the
    waveform crosses the midpoint values. From the midpoint values the
    alogrithm searches backward and forward until the voltage is below/above
    the lower/upper thresholds. Each contiguous such region is considered to be
    a half cycle. The half cycles are then used to find the full cycles,
    pretrial region , etc.    

    Returns: mask_dict = a dictionary masks created from the stimulus voltage
    with the following keys.

    pretrial            = mask for pretrial region
    trial               = mask for trial region 
    cycles              = list of masks for each cycle in the data
    half_cycles         = list of masks for each half cycle in the data
    even_half_cycles    = list of masks for each even half cycle in the data
    odd_half_cycles     = list of masks for eac odd half cycle in the data 

    """

    # Get list of masks for each half cycle region
    half_cycle_mask_list = find_half_cycles(
            t,
            volt,
            lower_threshold, 
            upper_threshold, 
            midpoint_window_width, 
            midpoint_window_min_len
            )

    # Get mask which is true during any cycle region - the 'logical or' of all half cycle masks
    all_half_cycles_mask = np.full(volt.shape, False, dtype=np.bool)
    for half_cycle_mask in half_cycle_mask_list:
        all_half_cycles_mask = np.logical_or(half_cycle_mask, all_half_cycles_mask)

    # Get pretrial and trial regions
    t_all_half_cycles = t[all_half_cycles_mask]
    trial_mask = t >= t_all_half_cycles[0]
    pretrial_mask = np.logical_not(trial_mask)

    # Get even and odd half cycle mask lists
    even_half_cycle_mask_list = [item for ind,item in enumerate(half_cycle_mask_list) if ind%2==0]
    odd_half_cycle_mask_list = [item for ind, item in enumerate(half_cycle_mask_list) if ind%2!=0]

    # Check that there are the same number of even and odd half cycles
    num_half_cycle_test = len(even_half_cycle_mask_list) == len(odd_half_cycle_mask_list)
    num_half_cycle_error_msg = 'number of even half cycels != number odd half cycles!!!'
    assert num_half_cycle_test, num_half_cycle_error_msg 

    # Create full cycle masks
    full_cycle_mask_list = []
    for even_mask, odd_mask in zip(even_half_cycle_mask_list, odd_half_cycle_mask_list):
        even_or_odd_mask = np.logical_or(even_mask, odd_mask)
        t_even_or_odd = t[even_or_odd_mask]
        full_cycle_mask = np.logical_and(t >= t_even_or_odd.min(), t<=t_even_or_odd.max())
        full_cycle_mask_list.append(full_cycle_mask)

    # Create dictionary of cycle masks
    mask_dict = {
            'pretrial'         : pretrial_mask,
            'trial'            : trial_mask,
            'cycles'           : full_cycle_mask_list,
            'half_cycles'      : half_cycle_mask_list,
            'even_half_cycles' : even_half_cycle_mask_list,
            'odd_half_cycles'  : odd_half_cycle_mask_list,
            }
    return mask_dict


def get_window_mask(x,window_value,window_width):
    """
    Returns a mask which is True when x is equal to the window_value +/-
    0.5*window_width.
    """
    window_mask_gt = x > window_value - 0.5*window_width
    window_mask_lt = x < window_value + 0.5*window_width
    window_mask = np.logical_and(window_mask_gt, window_mask_lt)
    return window_mask


def get_midpoint_mask(x,midpoint_width):
    """
    Returns a mask which is True when the x approximately equal to the midpoint
    value where the approximately is determined by the value of midpoint_width.  
    """
    min_x = x.min()
    max_x = x.max()
    midpoint_x = 0.5*(max_x + min_x)
    midpoint_mask = get_window_mask(x, midpoint_x, midpoint_width)
    return midpoint_mask


def find_contiguous_regions(mask,value):
    """
    Find contiguous regions where mask is equal to value. Returns list of masks for 
    contiguous regions.
    """
    contiguous_list = []
    region_mask = None 
    for ind, item in enumerate(mask):
        if item == value:
            if region_mask is None:
                region_mask = np.full(mask.shape,False,dtype=type(value))
            region_mask[ind] = True
        else:
            if region_mask is not None:
                contiguous_list.append(region_mask)
                region_mask = None
    return contiguous_list


def find_region_about_midpoint(volt, midpoint_ind, lower_threshold, upper_threshold):
    """
    Get mask for region about the midpoint region which is greater than lower_threshold and
    less than upper threshold.
    """
    region_mask = np.full(volt.shape,False,dtype=np.bool)

    # Search backward until upper or lower threshold
    ind = midpoint_ind[-1]
    while lower_threshold <= volt[ind] and  volt[ind] <= upper_threshold: 
        region_mask[ind] = True
        ind -= 1
        if ind < 0:
            break

    # Search forward until upper or lower threshold
    ind = midpoint_ind[0]
    while lower_threshold <= volt[ind] and  volt[ind] <= upper_threshold: 
        region_mask[ind] = True
        ind += 1
        if ind >= volt.shape[0]:
            break

    return region_mask


def find_half_cycles(t, volt, lower_threshold, upper_threshold, midpoint_window_width, midpoint_window_min_len):
    """
    Find the rising/falling half cycle regions of the voltage trace. 
    """

    # Get list of masks for contiguous mid point regions.
    all_midpoints_mask = get_midpoint_mask(volt, midpoint_window_width)
    midpoint_mask_list = find_contiguous_regions(all_midpoints_mask,True)
    midpoint_mask_list = [item for item in midpoint_mask_list if len(item) > midpoint_window_min_len]

    # Find maximum region about each midpoint with values between lower and
    # upper thresholds.  Each of these regions is a half cycle.
    half_cycle_mask_list = []
    ind = np.arange(volt.shape[0])
    for midpoint_mask in midpoint_mask_list:
        midpoint_ind = ind[midpoint_mask]
        half_cycle_mask =  find_region_about_midpoint(volt, midpoint_ind, lower_threshold, upper_threshold)
        half_cycle_mask_list.append(half_cycle_mask)

    return half_cycle_mask_list


def unwrap(x,discont=10):
    """
    Unwrap x values.
    """
    adjval = 0
    x_unwrap = np.zeros(x.shape)
    x_unwrap[0] = x[0]
    for i in range(1,x.shape[0]):
        dx = x[i] + adjval - x_unwrap[i-1]
        if dx < -0.5*discont:
            adjval += discont
        if dx > 0.5*discont:
            adjval -= discont
        x_unwrap[i] = x[i] + adjval
    return x_unwrap
