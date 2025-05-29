import numpy as np
from collections import deque

def update_max_energy(current_energy, max_energy, decay_factor, min_max_value, iteration_counter, reset_interval, debug=False):
    """
    Update the maximum energy value with decay over time and periodic resets.
    
    Args:
        current_energy (float): Current energy value to compare against max
        max_energy (float): Current maximum energy value
        decay_factor (float): How quickly the max decays (closer to 1 = slower decay)
        min_max_value (float): Lower bound for the max value
        iteration_counter (int): Current iteration counter
        reset_interval (int): Reset max energy to min every n iterations
        debug (bool): Enable debug printing
        
    Returns:
        tuple: (new_max_energy, new_iteration_counter)
    """
    # Increment iteration counter
    iteration_counter += 1
    
    # Check if it's time for a periodic reset
    if iteration_counter >= reset_interval:
        max_energy = min_max_value
        iteration_counter = 0
        if debug:
            print(f"[AudioUtils] Periodic reset: max energy reset to {min_max_value}")
    
    # Update max if current energy is higher
    if current_energy > max_energy:
        max_energy = current_energy
        if debug:
            print(f"[AudioUtils] New max energy: {max_energy:.4f}")
    
    # Apply decay to the max value
    max_energy *= decay_factor
    
    # Ensure max doesn't go below minimum threshold
    if max_energy < min_max_value:
        max_energy = min_max_value
        
    return max_energy, iteration_counter

def calculate_energy_percentage(current_energy, max_energy):
    """
    Calculate the percentage of the current energy relative to the maximum.
    
    Args:
        current_energy (float): Current energy value
        max_energy (float): Maximum energy value
        
    Returns:
        float: Percentage of max energy (0-100)
    """
    if max_energy <= 0:
        return 0.0
        
    percentage = (current_energy / max_energy) * 100.0
    # Clamp to 0-100 range
    return max(0.0, min(100.0, percentage))

def get_pipe_index_from_percentage(percentage, num_pipes, percentage_thresholds):
    """
    Map energy percentage to pipe index.
    
    Args:
        percentage (float): Energy percentage (0-100)
        num_pipes (int): Number of available pipes
        percentage_thresholds (list): List of percentage thresholds
        
    Returns:
        int: Pipe index
    """
    for i in range(len(percentage_thresholds) - 1):
        if percentage_thresholds[i] <= percentage < percentage_thresholds[i + 1]:
            return num_pipes - 1 - i
    
    # Anything above the highest threshold maps to the lowest pipe index
    return 0

def average_frequency_bins(frequency_data, custom_bin_ranges, rolling_window, debug=False):
    """
    Average frequency bins using custom ranges instead of equal-sized buckets.
    
    Args:
        frequency_data (np.ndarray or list): Array of frequency bin values
        custom_bin_ranges (list): List of tuples defining custom ranges: [(start, end), ...]
        rolling_window (deque): Rolling window for percentage change calculation
        debug (bool): Enable debug printing
        
    Returns:
        tuple: (averaged_bins, percentage_changes)
    """
    # Convert input to numpy array if it isn't already
    frequency_data = np.array(frequency_data)
    
    if debug:
        print(f"[AudioUtils] Input frequency data length: {len(frequency_data)}")
        print(f"[AudioUtils] Custom bin ranges: {custom_bin_ranges}")
        
    # Process each custom range
    averaged_bins = []
    
    for i, (start, end) in enumerate(custom_bin_ranges):
        # Ensure ranges are within bounds
        start = max(0, start)
        end = min(len(frequency_data) - 1, end)
        
        if start <= end:
            # Extract the slice and calculate median
            bin_slice = frequency_data[start:end+1]  # +1 because end is inclusive
            bin_average = np.mean(bin_slice)
            averaged_bins.append(bin_average)
            
            if debug:
                print(f"[AudioUtils] Bin {i}: range [{start}:{end+1}] -> mean = {bin_average:.4f}")
        else:
            # Invalid range, append 0
            averaged_bins.append(0.0)
            if debug:
                print(f"[AudioUtils] Bin {i}: invalid range [{start}:{end+1}] -> 0.0")
    
    averaged_result = np.array(averaged_bins)
    
    # Calculate percentage change if we have previous values
    percentage_changes = []
    if rolling_window:
        # Calculate average of rolling window for each bin
        window_array = np.array(list(rolling_window))
        window_averages = np.mean(window_array, axis=0)
        
        for i, (current, window_avg) in enumerate(zip(averaged_result, window_averages)):
            if window_avg != 0:
                pct_change = ((current - window_avg) / window_avg) * 100
            else:
                pct_change = 0.0 if current == 0 else float('inf')
            percentage_changes.append(pct_change)
    else:
        # First iteration, no previous values
        percentage_changes = [0.0] * len(averaged_result)
    
    if debug:
        print(f"[AudioUtils] Processed {len(frequency_data)} bins into {len(averaged_result)} custom bins")
        print(f"[AudioUtils] Current values: {averaged_result}")
        print(f"[AudioUtils] Percentage changes: {[f'{change:+.1f}%' for change in percentage_changes]}")
        
        # Print detailed comparison if we have previous values
        if rolling_window:
            window_array = np.array(list(rolling_window))
            window_averages = np.mean(window_array, axis=0)
            print(f"[AudioUtils] Rolling window size: {len(rolling_window)}")
            print("[AudioUtils] Detailed comparison (vs rolling window average):")
            for i, (current, window_avg, pct_change) in enumerate(zip(averaged_result, window_averages, percentage_changes)):
                direction = "📈" if pct_change > 0 else "📉" if pct_change < 0 else "➡️"
                print(f"  Bin {i}: {window_avg:.4f} (avg) → {current:.4f} ({pct_change:+.1f}%) {direction}")
        
    return averaged_result, percentage_changes

def get_volume_from_ft(frequency_data, method='rms', debug=False):
    """
    Calculate volume/amplitude from frequency transform data.
    
    Args:
        frequency_data (np.ndarray): Frequency domain data (magnitudes)
        method (str): Method to use - 'rms', 'peak', 'sum', 'weighted'
        debug (bool): Enable debug printing
        
    Returns:
        float: Volume level (0.0 to 1.0 range)
    """
    if frequency_data is None or len(frequency_data) == 0:
        return 0.0
        
    frequency_data = np.array(frequency_data)
    
    if method == 'rms':
        # Root Mean Square - most accurate for perceived loudness
        volume = np.sqrt(np.mean(frequency_data ** 2))
        
    elif method == 'peak':
        # Peak amplitude - maximum value
        volume = np.max(frequency_data)
        
    elif method == 'sum':
        # Sum of all magnitudes
        volume = np.sum(frequency_data)
        
    elif method == 'weighted':
        # Weighted by frequency (emphasizes mid-range frequencies)
        # Create weights that emphasize human hearing range (1kHz-4kHz)
        weights = np.ones_like(frequency_data)
        mid_start = len(frequency_data) // 4
        mid_end = len(frequency_data) // 2
        weights[mid_start:mid_end] *= 2.0  # Boost mid frequencies
        
        weighted_data = frequency_data * weights
        volume = np.sqrt(np.mean(weighted_data ** 2))
        
    else:
        # Default to RMS
        volume = np.sqrt(np.mean(frequency_data ** 2))
    
    # Normalize to 0-1 range (you may need to adjust this based on your data)
    volume = np.clip(volume, 0.0, 1.0)
    
    if debug:
        print(f"[AudioUtils] Volume ({method}): {volume:.4f}")
        
    return volume

def get_perceived_loudness(frequency_data, debug=False):
    """
    Calculate perceived loudness using A-weighting approximation.
    This better matches human hearing perception.
    
    Args:
        frequency_data (np.ndarray): Frequency domain data
        debug (bool): Enable debug printing
        
    Returns:
        float: Perceived loudness level
    """
    if frequency_data is None or len(frequency_data) == 0:
        return 0.0
        
    frequency_data = np.array(frequency_data)
    
    # Simple A-weighting approximation for frequency bins
    # This emphasizes frequencies around 2-4kHz where human hearing is most sensitive
    num_bins = len(frequency_data)
    frequencies = np.linspace(0, num_bins, num_bins)
    
    # Rough A-weighting curve (simplified)
    a_weights = np.ones_like(frequencies)
    
    # Boost mid frequencies (around bin 20-40 for typical 50-bin FFT)
    mid_range = slice(num_bins//4, num_bins//2)
    a_weights[mid_range] *= 3.0
    
    # Reduce low frequencies
    low_range = slice(0, num_bins//8)
    a_weights[low_range] *= 0.3
    
    # Reduce very high frequencies
    high_range = slice(3*num_bins//4, num_bins)
    a_weights[high_range] *= 0.7
    
    # Apply weighting and calculate RMS
    weighted_data = frequency_data * a_weights
    loudness = np.sqrt(np.mean(weighted_data ** 2))
    
    if debug:
        print(f"[AudioUtils] Perceived loudness: {loudness:.4f}")
        
    return np.clip(loudness, 0.0, 1.0)

def get_volume_quintile(frequency_data, method='rms', use_perceived_loudness=False, debug=False):
    """
    Calculate volume from frequency data and map it to quintiles (0-4) for pipe selection.
    
    Args:
        frequency_data (np.ndarray): Frequency domain data
        method (str): Volume calculation method ('rms', 'peak', 'sum', 'weighted')
        use_perceived_loudness (bool): Use perceived loudness instead of raw volume
        debug (bool): Enable debug printing
        
    Returns:
        int: Quintile index (0-4) representing volume level
    """
    if use_perceived_loudness:
        volume = get_perceived_loudness(frequency_data, debug)
    else:
        volume = get_volume_from_ft(frequency_data, method, debug)
    
    # Map volume (0.0-1.0) to quintiles (0-4)
    # Volume ranges: 0-0.2=0, 0.2-0.4=1, 0.4-0.6=2, 0.6-0.8=3, 0.8-1.0=4
    if volume < 0.2:
        quintile = 0
    elif volume < 0.4:
        quintile = 1
    elif volume < 0.6:
        quintile = 2
    elif volume < 0.8:
        quintile = 3
    else:
        quintile = 4
        
    if debug:
        volume_type = "perceived loudness" if use_perceived_loudness else f"volume ({method})"
        print(f"[AudioUtils] {volume_type.title()}: {volume:.4f} → Quintile: {quintile}")
        
    return quintile

def get_adaptive_volume_quintile(frequency_data, max_energy, decay_factor, min_max_value, 
                                iteration_counter, reset_interval, method='rms', 
                                use_perceived_loudness=False, debug=False):
    """
    Calculate volume quintile with adaptive thresholds based on recent volume history.
    This provides more dynamic response to volume changes.
    
    Args:
        frequency_data (np.ndarray): Frequency domain data
        max_energy (float): Current maximum energy value
        decay_factor (float): Energy decay factor
        min_max_value (float): Minimum energy value
        iteration_counter (int): Current iteration counter
        reset_interval (int): Reset interval for max energy
        method (str): Volume calculation method
        use_perceived_loudness (bool): Use perceived loudness instead of raw volume
        debug (bool): Enable debug printing
        
    Returns:
        tuple: (quintile, new_max_energy, new_iteration_counter)
    """
    if use_perceived_loudness:
        volume = get_perceived_loudness(frequency_data, debug)
    else:
        volume = get_volume_from_ft(frequency_data, method, debug)
    
    # Update max energy tracking
    max_energy, iteration_counter = update_max_energy(
        volume, max_energy, decay_factor, min_max_value, 
        iteration_counter, reset_interval, debug
    )
    
    # Calculate percentage of max volume
    if max_energy > 0:
        volume_percentage = (volume / max_energy) * 100.0
    else:
        volume_percentage = 0.0
        
    # Map percentage to quintiles
    if volume_percentage < 20:
        quintile = 0
    elif volume_percentage < 40:
        quintile = 1
    elif volume_percentage < 60:
        quintile = 2
    elif volume_percentage < 80:
        quintile = 3
    else:
        quintile = 4
        
    if debug:
        volume_type = "perceived loudness" if use_perceived_loudness else f"volume ({method})"
        print(f"[AudioUtils] {volume_type.title()}: {volume:.4f}")
        print(f"[AudioUtils] Max volume: {max_energy:.4f}")
        print(f"[AudioUtils] Volume percentage: {volume_percentage:.1f}% → Quintile: {quintile}")
        
    return quintile, max_energy, iteration_counter 

def hz_to_mel(hz):
    """Convert frequency in Hz to mel scale."""
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    """Convert mel scale back to frequency in Hz."""
    return 700 * (10**(mel / 2595) - 1)

def convert_to_mel_bins(frequency_data, sample_rate=44100, n_fft_bins=50, max_freq=10000, min_freq=201, n_mel_bins=5, filter_method='overlap', debug=False):
    """
    Convert linear FFT bins to mel-spaced perceptual frequency bins.
    
    Args:
        frequency_data (np.ndarray or list): Array of 50 linear FFT bin values (0-10kHz)
        sample_rate (int): Sample rate used for FFT (default 22050 Hz)
        n_fft_bins (int): Number of input FFT bins (should be 50)
        max_freq (int): Maximum frequency covered by FFT bins (default 10000 Hz)
        min_freq (int): Minimum frequency to include (default 201 Hz to exclude bin0)
        n_mel_bins (int): Number of output mel bins (default 5)
        filter_method (str): Filtering method - 'overlap' (include bins with any overlap above min_freq) 
                           or 'center' (include only bins whose center frequency is above min_freq)
        debug (bool): Enable debug printing
        
    Returns:
        tuple: (mel_bins, mel_frequencies, bin_mapping)
            mel_bins: Array of mel-spaced frequency bin energies
            mel_frequencies: Center frequencies of each mel bin in Hz
            bin_mapping: List showing which linear bins map to each mel bin
    """
    frequency_data = np.array(frequency_data)
    
    if len(frequency_data) != n_fft_bins:
        if debug:
            print(f"[MelUtils] Warning: Expected {n_fft_bins} bins, got {len(frequency_data)}")
    
    # Calculate frequency resolution (Hz per bin)
    freq_resolution = max_freq / n_fft_bins  # 10000 Hz / 50 bins = 200 Hz per bin
    
    # Create frequency array for each linear bin (center frequencies)
    linear_freqs = np.linspace(freq_resolution/2, max_freq - freq_resolution/2, n_fft_bins)
    
    # Convert frequency range to mel scale, starting from min_freq
    min_mel = hz_to_mel(min_freq)
    # max_mel = hz_to_mel(max_freq - freq_resolution/2)  # End at last bin center
    max_mel = hz_to_mel(max_freq)  # Use full max frequency range
    
    # Create mel-spaced frequency points
    mel_points = np.linspace(min_mel, max_mel, n_mel_bins + 1)
    mel_freqs = mel_to_hz(mel_points)
    
    # Calculate center frequencies for each mel bin
    mel_centers = []
    for i in range(n_mel_bins):
        center_freq = (mel_freqs[i] + mel_freqs[i + 1]) / 2
        mel_centers.append(center_freq)
    
    # Map linear bins to mel bins
    mel_bins = []
    bin_mapping = []
    
    for i in range(n_mel_bins):
        # Find which linear bins fall into this mel bin range
        start_freq = mel_freqs[i]
        end_freq = mel_freqs[i + 1]
        
        # Find linear bins that overlap with this mel bin
        bin_indices = []
        bin_weights = []
        
        for j, lin_freq in enumerate(linear_freqs):
            # Check if this linear bin overlaps with current mel bin
            bin_start = lin_freq - freq_resolution/2
            bin_end = lin_freq + freq_resolution/2
            
            # Apply frequency filtering based on chosen method
            if filter_method == 'center':
                # Center frequency filtering: Skip bins whose center frequency is below min_freq
                if lin_freq < min_freq:
                    continue
            else:  # filter_method == 'overlap' (default)
                # Overlap filtering: Skip bins that are entirely below min_freq
                if bin_end <= min_freq:
                    continue
            
            # Adjust bin_start if it's below min_freq (for overlap method)
            if filter_method == 'overlap' and bin_start < min_freq:
                bin_start = min_freq
            
            # Calculate overlap
            overlap_start = max(start_freq, bin_start)
            overlap_end = min(end_freq, bin_end)
            
            if overlap_end > overlap_start:
                # There's overlap - calculate weight based on overlap amount
                overlap_amount = overlap_end - overlap_start
                total_bin_width = bin_end - max(bin_start, min_freq)  # Adjust for min_freq clipping
                
                if total_bin_width > 0:
                    weight = overlap_amount / total_bin_width
                    bin_indices.append(j)
                    bin_weights.append(weight)
        
        # Calculate weighted average for this mel bin
        if bin_indices:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for bin_idx, weight in zip(bin_indices, bin_weights):
                weighted_sum += frequency_data[bin_idx] * weight
                total_weight += weight
            
            if total_weight > 0:
                mel_bin_value = weighted_sum / total_weight
            else:
                mel_bin_value = 0.0
                
            bin_mapping.append(list(zip(bin_indices, bin_weights)))
        else:
            mel_bin_value = 0.0
            bin_mapping.append([])
        
        mel_bins.append(mel_bin_value)
    
    mel_bins = np.array(mel_bins)
    
    if debug:
        print(f"[MelUtils] Linear frequency resolution: {freq_resolution:.1f} Hz per bin")
        print(f"[MelUtils] Frequency range: {min_freq} - {max_freq} Hz")
        print(f"[MelUtils] Mel frequency ranges:")
        for i in range(n_mel_bins):
            start_hz = mel_freqs[i]
            end_hz = mel_freqs[i + 1]
            center_hz = mel_centers[i]
            print(f"  Mel bin {i}: {start_hz:.0f}-{end_hz:.0f} Hz (center: {center_hz:.0f} Hz)")
            
            # Show which linear bins contribute
            if bin_mapping[i]:
                contributing_bins = [f"bin{idx}({weight:.2f})" for idx, weight in bin_mapping[i]]
                print(f"    Linear bins: {', '.join(contributing_bins)}")
        
        print(f"[MelUtils] Mel bin energies: {mel_bins}")
        print(f"[MelUtils] Most energetic mel bin: {np.argmax(mel_bins)} (energy: {np.max(mel_bins):.4f})")
    
    return mel_bins, mel_centers, bin_mapping

def get_perceptual_frequency_ranges():
    """
    Return standard perceptual frequency ranges for audio analysis.
    
    Returns:
        dict: Dictionary with frequency range names and their Hz ranges
    """
    return {
        'sub_bass': (20, 60),      # Sub-bass/rumble
        'bass': (60, 250),         # Bass/kick drums
        'low_mids': (250, 500),    # Low midrange/warmth
        'mids': (500, 2000),       # Midrange/vocals
        'high_mids': (2000, 4000), # High midrange/presence
        'treble': (4000, 8000),    # Treble/brightness
        'air': (8000, 20000)       # Air/sparkle (beyond our 8kHz limit)
    }

def convert_to_decibels(energy_values, reference_energy=1e-10, min_db=-80.0, db_offset=0.0, debug=False):
    """
    Convert energy values to decibels (dB) for perceptual audio analysis.
    
    This function converts linear energy values to a logarithmic decibel scale,
    which better represents human perception of loudness and is standard practice
    in audio signal processing.
    
    Args:
        energy_values (np.ndarray or list): Linear energy values to convert
        reference_energy (float): Reference energy level (prevents log(0), default 1e-10)
        min_db (float): Minimum dB value to clamp results (default -80 dB)
        db_offset (float): Offset to add to dB values (e.g., 80.0 to make values mostly positive)
        debug (bool): Enable debug printing
        
    Returns:
        np.ndarray: Energy values converted to decibels
    """
    energy_values = np.array(energy_values)
    
    # Ensure we don't take log of zero or negative values
    # Add small reference energy to avoid log(0)
    safe_energies = np.maximum(energy_values, reference_energy)
    
    # Convert to decibels: dB = 10 * log10(energy)
    # Using 10*log10 for power/energy, not 20*log10 which is for amplitude
    db_values = 10 * np.log10(safe_energies)
    
    # Add offset if specified (e.g., to make values mostly positive)
    db_values += db_offset
    
    # Clamp to minimum dB to avoid extreme negative values
    db_values = np.maximum(db_values, min_db + db_offset)
    
    if debug:
        print(f"[dBUtils] Energy range: {np.min(energy_values):.2e} to {np.max(energy_values):.2e}")
        print(f"[dBUtils] dB range: {np.min(db_values):.1f} to {np.max(db_values):.1f} dB")
        print(f"[dBUtils] Reference energy: {reference_energy:.2e}")
        print(f"[dBUtils] Min dB clamp: {min_db + db_offset} dB")
        print(f"[dBUtils] dB offset applied: {db_offset} dB")
        
        # Show conversion details for each bin
        for i, (energy, db) in enumerate(zip(energy_values, db_values)):
            print(f"  Bin {i}: {energy:.4f} → {db:.1f} dB")
    
    return db_values 