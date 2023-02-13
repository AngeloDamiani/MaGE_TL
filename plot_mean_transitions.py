from stats import read_errors, plot_errors

folder = "logs"
prefix = "G1_"
black_list = []

errors = read_errors(folder, prefix)
plot_errors(errors, black_list)
