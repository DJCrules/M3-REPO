import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Graphing/industrial_trend.csv')  # Replace 'your_data.csv' with the path to your actual CSV file

data['Date'] = pd.to_datetime(data['Date'])

plt.figure(figsize=(10, 6))  # Optional: Adjust figure size
plt.plot(data['Date'], data['value'], label='Value', color='b', marker='o')

plt.title('Non Domestic Birmingham Power Consumption')
plt.xlabel('Date')
plt.ylabel('Non Domestic TWh')

plt.grid(True)

plt.legend()

plt.xticks(rotation=45)  # Rotate x-axis labels for better readability (optional)
plt.tight_layout()  # Adjust layout to fit everything
plt.show()

# Save the plot as an image (optional)
# plt.savefig('plot.png')
