# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('airline8.csv')

df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month

# Here I am appying Fourier Transform
DataOfPassengers = df['Number']
resultFXT = np.fft.fft(DataOfPassengers)
FrequencyOfData = np.fft.fftfreq(len(DataOfPassengers))
power_spectrum = np.abs(resultFXT)**2

# Plot the Power Spectrum
plt.figure(figsize=(10, 6))
plt.plot(FrequencyOfData, power_spectrum)
plt.xlabel('Frequency (1/day)')
plt.ylabel('Power')
plt.title(f'Fourier Power Spectrum (ID: 23121983)')
plt.show()

# Average Daily Passengers by Month
monthly_avg = df.groupby('Month')['Number'].mean()

# Plot Average Daily Passengers by Month
plt.figure(figsize=(10, 6))
monthly_avg.plot(kind='bar')
plt.xlabel('Month')
plt.ylabel('Average Daily Passengers')
plt.title(f'Average Daily Passengers by Month (ID: 23121983)')
plt.xticks(ticks=np.arange(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
plt.show()

# Fourier Series Approximation (First 8 Terms)
n_terms = 8
fourier_coeffs = resultFXT[:n_terms]
reconstructed_data = np.fft.ifft(fourier_coeffs, n=len(DataOfPassengers))

# Plot Original Data and Fourier Series Approximation
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], DataOfPassengers, label='Original Data')
plt.plot(df['Date'], reconstructed_data.real, label='Fourier Approximation (8 Terms)', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Passenger Count')
plt.title(f'Fourier Series Approximation (ID: 23121983)')
plt.legend()
plt.show()

# Calculate X and Y (Summer Revenue and Passenger Fractions)
summer_months = df[df['Month'].isin([6, 7, 8])]

# Calculate total revenue and passengers for summer months
total_revenue = df['Price'].sum() * df['Number'].sum()  # Assuming price is per passenger
summer_revenue = summer_months['Price'].sum() * summer_months['Number'].sum()

# Calculate total number of passengers
total_passengers = df['Number'].sum()
summer_passengers = summer_months['Number'].sum()

# Calculate X and Y
X = (summer_revenue / total_revenue) * 100
Y = (summer_passengers / total_passengers) * 100

# Print X and Y
print(f"X (Summer Revenue Percentage): {X:.2f}%")
print(f"Y (Summer Passengers Percentage): {Y:.2f}%")