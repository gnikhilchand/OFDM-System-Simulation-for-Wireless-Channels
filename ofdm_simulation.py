import numpy as np
import matplotlib.pyplot as plt

# --- OFDM System Parameters ---
K = 64              # Number of subcarriers
CP = K // 4         # Length of the cyclic prefix (25% of the number of subcarriers)
P = 8               # Number of pilot subcarriers
pilot_value = 3+3j  # The known value for pilot carriers
all_carriers = np.arange(K)  # Indices of all subcarriers
pilot_carriers = all_carriers[::K//P] # Indices of pilot subcarriers, spaced apart
data_carriers = np.delete(all_carriers, pilot_carriers) # Indices of data subcarriers

# --- Simulation Parameters ---
MODULATION_SCHEME = 'QPSK'
NUM_SYMBOLS = 100 # Number of OFDM symbols to transmit
SNR_dB = 20 # Signal to Noise Ratio in dB

def qpsk_modulate(bits):
    """ Modulates a bit stream to QPSK symbols. """
    # Reshape bits into pairs and map to complex symbols
    # 00->(1+j)/sqrt(2), 01->(-1+j)/sqrt(2), 10->(1-j)/sqrt(2), 11->(-1-j)/sqrt(2)
    symbols = (1 - 2 * bits[0::2]) + 1j * (1 - 2 * bits[1::2])
    return symbols / np.sqrt(2)

def qpsk_demodulate(symbols):
    """ Demodulates QPSK symbols back to a bit stream. """
    bits_real = (np.real(symbols) < 0).astype(int)
    bits_imag = (np.imag(symbols) < 0).astype(int)
    bits = np.empty(len(symbols)*2, dtype=int)
    bits[0::2] = bits_real
    bits[1::2] = bits_imag
    return bits

def channel(signal, snr_db, channel_response):
    """ 
    Simulates a multipath channel and adds AWGN.
    - signal: The transmitted signal.
    - snr_db: The desired Signal-to-Noise ratio in dB.
    - channel_response: The impulse response of the multipath channel.
    """
    # Apply multipath fading
    convolved = np.convolve(signal, channel_response, mode='full')
    # Normalize signal power to 1 after convolution
    signal_power = np.mean(np.abs(convolved**2))
    
    # Calculate noise power based on SNR
    noise_power = signal_power / (10**(snr_db / 10))
    
    # Generate complex Gaussian noise
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*convolved.shape) + 1j*np.random.randn(*convolved.shape))
    
    return convolved + noise

def ofdm_demodulate(received_signal, channel_est):
    """ Performs OFDM demodulation including FFT and equalization. """
    # --- CP Removal ---
    signal_without_cp = received_signal[CP:(CP+K)]
    
    # --- FFT ---
    symbols_freq_domain = np.fft.fft(signal_without_cp)
    
    # --- Channel Equalization ---
    equalized_symbols = symbols_freq_domain / channel_est
    
    return equalized_symbols[data_carriers] # Return only data symbols

# --- Main Simulation Logic ---
# Define a simple 3-tap multipath channel
channel_response = np.array([1, 0.5, 0.2])
H_exact = np.fft.fft(channel_response, K) # True frequency response

# Generate bits for all OFDM symbols
bits_per_symbol = len(data_carriers) * 2 # 2 bits per QPSK symbol
total_bits = np.random.randint(0, 2, NUM_SYMBOLS * bits_per_symbol)
bits_reshaped = total_bits.reshape((NUM_SYMBOLS, bits_per_symbol))

# Transmitted and received symbol storage for plotting
all_tx_symbols = []
all_rx_symbols = []
all_equalized_symbols = []

for i in range(NUM_SYMBOLS):
    # --- Transmitter ---
    bits = bits_reshaped[i]
    
    # 1. Modulate data bits to QPSK symbols
    qpsk_symbols = qpsk_modulate(bits)
    all_tx_symbols.extend(qpsk_symbols)
    
    # 2. Map data and pilots to subcarriers
    ofdm_data = np.zeros(K, dtype=complex)
    ofdm_data[pilot_carriers] = pilot_value
    ofdm_data[data_carriers] = qpsk_symbols
    
    # 3. IFFT: Convert frequency-domain symbols to time-domain signal
    ofdm_time_domain_signal = np.fft.ifft(ofdm_data)
    
    # 4. Add Cyclic Prefix (CP)
    ofdm_with_cp = np.concatenate((ofdm_time_domain_signal[-CP:], ofdm_time_domain_signal))
    
    # --- Channel ---
    received_signal_with_channel = channel(ofdm_with_cp, SNR_dB, channel_response)
    
    # --- Receiver ---
    # Perfect channel estimation for simplicity (using the known true response)
    # In a real system, H would be estimated from the pilot symbols.
    
    # 1. Demodulate the received signal
    demodulated_data_symbols = ofdm_demodulate(received_signal_with_channel, H_exact)
    
    # Store symbols for visualization
    # To get symbols before equalization, we do FFT but don't divide by H
    symbols_before_eq = np.fft.fft(received_signal_with_channel[CP:(CP+K)])[data_carriers]
    all_rx_symbols.extend(symbols_before_eq)
    all_equalized_symbols.extend(demodulated_data_symbols)


# --- Plotting Constellations ---
plt.figure(figsize=(12, 6))
plt.suptitle(f'OFDM Constellation Diagrams at SNR = {SNR_dB} dB', fontsize=16)

# Plot 1: Received symbols before equalization
plt.subplot(1, 2, 1)
plt.scatter(np.real(all_rx_symbols), np.imag(all_rx_symbols), alpha=0.5)
plt.title("Before Channel Equalization")
plt.xlabel("In-Phase")
plt.ylabel("Quadrature")
plt.grid(True)
plt.axis('equal')

# Plot 2: Received symbols after equalization
plt.subplot(1, 2, 2)
plt.scatter(np.real(all_equalized_symbols), np.imag(all_equalized_symbols), alpha=0.5)
plt.title("After Channel Equalization")
plt.xlabel("In-Phase")
plt.grid(True)
plt.axis('equal')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()