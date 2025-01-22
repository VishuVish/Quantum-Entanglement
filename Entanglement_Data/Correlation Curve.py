import matplotlib.pyplot as plt

# Define a function to read the file
def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()[4:]

    data = {"Y": [], "X=0": [], "X=45": [], "X=90": [], "X=135": []}
    for line in lines: # Skip the header line
        values = line.strip().split()
        data["Y"].append(float(values[0]))
        data["X=0"].append(float(values[1]))
        data["X=45"].append(float(values[2]))
        data["X=90"].append(float(values[3]))
        data["X=135"].append(float(values[4]))

    return data

# Read the data from the file
file_path = 'corr_meas_02.txt'  # Replace with the path to your text file
data = read_data(file_path)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(data["Y"], data["X=0"], label="H")
plt.plot(data["Y"], data["X=45"], label="-")
plt.plot(data["Y"], data["X=90"], label="V")
plt.plot(data["Y"], data["X=135"], label="+")

# Adding labels and title
plt.xlabel("Polarization Angles in °")
plt.ylabel("Counts")
plt.title("Correlation Curve for product state")
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
