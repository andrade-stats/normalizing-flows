
import numpy
import torch
import matplotlib.pyplot as plt
import new_flows

print("PyTorch - Version = ", torch.__version__)
print("numpy - Version = ", numpy.__version__)

LARGE_SIZE = 25
MEDIUM_SIZE = 20


# plot_name = "LOFT"
plot_name = "Asymmetric Soft Clamp"

fig, all_subplots = plt.subplots(1, 1, figsize=(15,6))
fig.suptitle(plot_name, fontsize = LARGE_SIZE)

if plot_name == "LOFT":
    # left_bound = -200.0
    # right_bound = 200.0
    left_bound = -14.0
    right_bound = 14.0

    x_values = torch.linspace(left_bound, right_bound, 1000)
    x_values_numpy = numpy.asarray(x_values.tolist())
    
    # all_subplots.set_xticks(numpy.arange(left_bound, right_bound + 1.0, 50.0))
    all_subplots.set_xticks(numpy.arange(left_bound, right_bound + 1.0, 2.0))

    all_subplots.set_yticks(numpy.arange(start = -5.0, stop =5.0 + 0.01, step = 1.0))
    all_subplots.set_ylim((-5.0, 5.0))

    plt.ylabel(r'$g(z)$', fontsize = MEDIUM_SIZE)
    plt.xlabel(r'$z$', fontsize = MEDIUM_SIZE)

    y_values, _ = new_flows.TrainableLOFTLayer.LOFT_forward_static(t = torch.tensor(2.0), z = x_values)
    y_values_numpy = numpy.asarray(y_values.tolist())
    all_subplots.plot(x_values_numpy, y_values_numpy)

    filename = "../NormalizingFlows_private/latex/IOP/IOP_final/" + "LOFT.png"

else:

    left_bound = -14.0
    right_bound = 14.0

    x_values = torch.linspace(left_bound, right_bound, 1000)
    x_values_numpy = numpy.asarray(x_values.tolist())
    
    all_subplots.set_xticks(numpy.arange(left_bound, right_bound + 1.0, 2))
    all_subplots.set_yticks(numpy.arange(start = -1.9, stop = 0.1 + 0.01, step = 0.2))
    all_subplots.set_ylim((-1.9, 0.2))

    plt.ylabel(r'$c(s)$', fontsize = MEDIUM_SIZE)
    plt.xlabel(r'$s$', fontsize = MEDIUM_SIZE)

    plt.grid(axis='x', color='0.95')
    plt.grid(axis='y', color='0.95')

    y_values = new_flows.softClampAsymAdvanced_differentImpl(x_values, negAlpha = 2.0, posAlpha = 0.1)
    y_values_numpy = numpy.asarray(y_values.tolist())
    all_subplots.plot(x_values_numpy, y_values_numpy)
    
    filename = "../NormalizingFlows_private/latex/IOP/IOP_final/" + "asym_soft_clamp.png"
    

all_subplots.xaxis.set_tick_params(labelsize=MEDIUM_SIZE)
all_subplots.yaxis.set_tick_params(labelsize=MEDIUM_SIZE)

plt.grid(axis='x', color='0.95')
plt.grid(axis='y', color='0.95')

# plt.show()
plt.savefig(filename)
