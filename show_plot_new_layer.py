
import numpy
import torch
import matplotlib.pyplot as plt
import new_flows

SMALL_SIZE = 12
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


# plot_name = "LOFT"
plot_name = "Asymmetric Soft Clamp"

fig, all_subplots = plt.subplots(1, 1, figsize=(15,6))
fig.suptitle(plot_name)
 
if plot_name == "LOFT":
    # left_bound = -200.0
    # right_bound = 200.0
    left_bound = -14.0
    right_bound = 14.0

    x_values = torch.linspace(left_bound, right_bound, 1000)
    print("x_values = ", x_values)

    # all_subplots.set_xticks(numpy.arange(left_bound, right_bound + 1.0, 50.0))
    all_subplots.set_xticks(numpy.arange(left_bound, right_bound + 1.0, 2.0))

    all_subplots.set_yticks(numpy.arange(start = -5.0, stop =5.0 + 0.01, step = 1.0))
    all_subplots.set_ylim((-5.0, 5.0))

    plt.ylabel(r'$g(z)$')
    plt.xlabel(r'$z$')

    y_values, _ = new_flows.TrainableLOFTLayer.LOFT_forward_static(t = torch.tensor(2.0), z = x_values)
    all_subplots.plot(x_values, y_values)

    plt.grid(axis='x', color='0.95')
    plt.grid(axis='y', color='0.95')

    # plt.show()
    plt.savefig("all_plots_final/" + "LOFT.png")
else:

    left_bound = -14.0
    right_bound = 14.0

    x_values = torch.linspace(left_bound, right_bound, 1000)
    print("x_values = ", x_values)

    all_subplots.set_xticks(numpy.arange(left_bound, right_bound + 1.0, 2))
    all_subplots.set_yticks(numpy.arange(start = -1.9, stop = 0.1 + 0.01, step = 0.2))
    all_subplots.set_ylim((-1.9, 0.2))

    plt.ylabel(r'$c(s)$')
    plt.xlabel(r'$s$')

    plt.grid(axis='x', color='0.95')
    plt.grid(axis='y', color='0.95')

    all_subplots.plot(x_values, new_flows.softClampAsymAdvanced_differentImpl(x_values, negAlpha = 2.0, posAlpha = 0.1))
    
    # plt.show()
    plt.savefig("all_plots_final/" + "asym_soft_clamp.png")



