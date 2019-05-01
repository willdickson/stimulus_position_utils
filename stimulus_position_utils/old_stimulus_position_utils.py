import numpy as np

def get_masks(volt,threshold=0.09):
    """
    Creates boolean masks for various regions of the stimulus position voltage. 
    Returns a dictionary of masks create with the following keys:

    pretrial            = mask for pretrial region
    trial               = mask for trial region 
    cycles              = list of masks for each cycle in the data
    even_half_cycles    = list of masks for each even half cycle in the data
    odd_half_cycles     = list of masks for eac odd half cycle in the data 

    """

    # Get masks for pts above and below voltage threshold
    above_mask = volt > threshold
    below_mask = np.logical_not(above_mask)

    # We need full cycles which begin and end below the threshold for this
    # procedure to work correctly. However, sometimes the last cycle maybe
    # truncated early.  As a quick fix for this issue if the last point in the
    # data set is not below the threshold we pretend that it is anyway to catch
    # the last cycle. 
    if volt[-1] > threshold:
        below_mask[-1] = True
    
    # Get indices for all pts below voltage threshold
    ind = np.arange(volt.shape[0])
    below_ind = ind[below_mask]
    
    # Difference and find steps greater larger than one to get half cycles
    diff_below_ind = np.zeros(below_ind.shape)
    diff_below_ind[:-1] = np.diff(below_ind)
    half_cycles_mask = diff_below_ind > 1
    ind_half_cycles = below_ind[half_cycles_mask]
    ind_half_cycles = np.append(ind_half_cycles,np.array(ind[-1]))
    
    # Get masks for pretrial and trial regions
    pretrial_mask = ind < ind_half_cycles[0]
    trial_mask = np.logical_not(pretrial_mask)
    
    # Get list of masks for half cycles
    even_half_cycle_mask_list =[]
    for n,m in zip(ind_half_cycles[:-2:2], ind_half_cycles[1::2]):
        even_half_cycle_mask = np.logical_and(ind >= n, ind < m)
        even_half_cycle_mask_list.append(even_half_cycle_mask)
    
    odd_half_cycle_mask_list =[]
    for n,m in zip(ind_half_cycles[1:-2:2], ind_half_cycles[2::2]):
        odd_half_cycle_mask = np.logical_and(ind >= n, ind < m)
        odd_half_cycle_mask_list.append(odd_half_cycle_mask)
    
    # Get list of masks for cycles
    cycle_mask_list = []
    ind_cycles = ind_half_cycles[0::2]
    for n,m in zip(ind_cycles[:-1], ind_cycles[1:]):
        cycle_mask = np.logical_and(ind >= n, ind < m)
        cycle_mask_list.append(cycle_mask)

    masks = {
            'pretrial'         : pretrial_mask,
            'trial'            : trial_mask,
            'cycles'           : cycle_mask_list,
            'even_half_cycles' : even_half_cycle_mask_list,
            'odd_half_cycles'  : odd_half_cycle_mask_list,
            'ind_half_cycles'  : ind_half_cycles,
            'above'            : above_mask,
            'below'            : below_mask,
            }
    return masks


