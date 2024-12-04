import GPUtil
import torch
import numpy

REPETITIONS_FOR_MC_ERROR = 20

INFO_STR = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_TYPE = None

NUM_SAMPLES_MCMC = 100000
NUM_WARMUP_MCMC = 100000
NUM_CHAINS_MCMC = 4
THINNING_MCMC = 10

ALL_RESULTS = "all_results/"
ALL_MODELS = "all_trained_models/"

# ALL_RESULTS = "all_results_new2/"
# ALL_MODELS = "all_trained_models_new/"

def get_model_filename():
    return ALL_MODELS + INFO_STR + "_q_model"

def get_model_filename_best():
    best_model_filename_str = ALL_MODELS + INFO_STR + "_q_model_best"
    print("best model filename = ", best_model_filename_str)
    return best_model_filename_str

def moveToDevice(obj):
    if DATA_TYPE == "double":
        obj = obj.double() # set double as default
    else:
        assert(DATA_TYPE == "float")
        obj = obj.float()
    obj = obj.to(device=DEVICE)
    return obj

def saveStatistics(obj, filenameSuffix):
    filename = ALL_RESULTS + INFO_STR + "_" + filenameSuffix
    numpy.save(filename, obj)
    print("successfully saved to ", filename)
    return

def loadStatistics(filenameSuffix, info_str = None):
    if info_str is not None:
        filename = ALL_RESULTS + info_str + "_" + filenameSuffix
    else:
        filename = ALL_RESULTS + INFO_STR + "_" + filenameSuffix
    return numpy.load(filename + ".npy", allow_pickle = True).item()

def loadArray(filenameSuffix):
    filename = ALL_RESULTS + INFO_STR + "_" + filenameSuffix
    return numpy.load(filename + ".npy", allow_pickle = True)


def getInfoStr_new(args, TARGET_NAME, DATA_DESCRIPTOR_STR, FLOW_TYPE, NUMBER_OF_FLOWS, MAX_ITERATION, NUM_SAMPLES, DIVERGENCE_FOR_OPTIMIZATION, CUSHION_TYPE, CUSHION_T, L2_STRENGTH, L2_CLIP_NORM, NR_CUSHIONS, INITIALIZATION, ANNEALING, LEARNING_RATE_EXP, NR_MIXTURE_COMPONENTS, LEARN_MIXTURE_WEIGHTS, realNVP_threshold, realNVP_variation, realNVP_spec, redGradVarEst, opt, scaleShiftLayer, trainable_base, use_student_base):

    target_name_short = "".join([c for c in TARGET_NAME if c.isupper()])
    assert(target_name_short == "MNM" or target_name_short == "MST" or target_name_short == "F" or target_name_short == "CLR" or target_name_short == "HPLR" or target_name_short == "STM" or target_name_short == "BL" or target_name_short == "HR")

    if target_name_short == "BL":
        target_name_short += "_" + str(args.lambd)

    if target_name_short == "MNM" or target_name_short == "STM":
        target_name_short += "_" + str(args.targetK) + "_" + str(args.targetMeanFac)

    # if target_name_short == "MST" or target_name_short == "F":
    #     target_name_short += "_" + str(args.var)
    #     print("target_name_short = ", target_name_short)
     
    if FLOW_TYPE == "GaussianOnly":
        assert(CUSHION_TYPE is None)
        basic_info_str = target_name_short + "_" + DATA_DESCRIPTOR_STR + "_" + FLOW_TYPE +  "_" + str(MAX_ITERATION) + "_" + str(NUM_SAMPLES) + "_" + str(DIVERGENCE_FOR_OPTIMIZATION) + "_" + str(NR_MIXTURE_COMPONENTS)  + "_" + str(LEARN_MIXTURE_WEIGHTS)
    else:    
        NO_ACT_NORM = (args.no_act_norm == "yes")
        basic_info_str = target_name_short + "_" + DATA_DESCRIPTOR_STR + "_" + FLOW_TYPE + "_" + str(NUMBER_OF_FLOWS) +  "_" + str(MAX_ITERATION) + "_" + str(NUM_SAMPLES) + "_" + str(DIVERGENCE_FOR_OPTIMIZATION) + "_" + str(CUSHION_TYPE)  + "_" + str(NO_ACT_NORM) + "_" + str(NR_MIXTURE_COMPONENTS)  + "_" + str(LEARN_MIXTURE_WEIGHTS)

    if DATA_TYPE == "float":
        basic_info_str += "_ft"
    else:
        assert(DATA_TYPE == "double")

    assert(ANNEALING == "yes" or ANNEALING == "no")
    if ANNEALING != "yes":
        basic_info_str += "_" + str(ANNEALING) + "ann"

    assert(LEARNING_RATE_EXP <= 5 and LEARNING_RATE_EXP >= 1)
    if LEARNING_RATE_EXP != 5:
        basic_info_str += "_" + str(LEARNING_RATE_EXP) + "LRE"

    assert(INITIALIZATION == "random" or INITIALIZATION == "zeros")
    if INITIALIZATION != "random":
        basic_info_str += "_" + str(INITIALIZATION) + "init"

    if CUSHION_TYPE == "LOFT" or CUSHION_TYPE == "TrainableLOFT" or CUSHION_TYPE == "ScaledTanh" or CUSHION_TYPE == "Truncation":
        assert(FLOW_TYPE != "GaussianOnly")
        basic_info_str += "_" + str(CUSHION_T) + "_" + str(NR_CUSHIONS)
    else:
        assert(CUSHION_TYPE == "none" or FLOW_TYPE == "GaussianOnly")
    
    if not (L2_STRENGTH == 0.0 and L2_CLIP_NORM is None):
        basic_info_str += "_" + str(L2_STRENGTH) + "_" + str(L2_CLIP_NORM)

    if realNVP_threshold is not None:
        basic_info_str += "_" + str(realNVP_threshold) + "th"

    if realNVP_variation is not None:
        basic_info_str += "_" + realNVP_variation

    if realNVP_spec == "MaskedAffineFlowSquashedSigmoid":
        basic_info_str += "_" + "sS"
    elif realNVP_spec == "MaskedAffineFlowSoftClamp":
        basic_info_str += "_" + "sC"
    else:
        assert(realNVP_spec is None)
    
    if redGradVarEst is not None:
        basic_info_str += "_" + redGradVarEst
    
    if opt != "Adam":
        basic_info_str += "_" + opt
    
    if scaleShiftLayer is not None:
        basic_info_str += "_" + scaleShiftLayer
    
    if trainable_base == "no":
        basic_info_str += "_nTB"
    
    if use_student_base == "yes":
        basic_info_str += "_ST"
    
    assert(len(ALL_RESULTS + basic_info_str) < 220)
    print("basic_info_str = ", basic_info_str)
    
    return basic_info_str


def getMCMC_spec_str(TARGET_NAME, DATA_DESCRIPTOR_STR, num_samples, num_warmup, num_chains, thinning, device = "cpu"):

    if TARGET_NAME == "MultivariateNormalMixture":
        import target_distributions
        dds = DATA_DESCRIPTOR_STR + "_" + str(3) + "_" + str(target_distributions.MultivariateNormalMixture.FACTOR_FOR_MEAN)
        print("dds = ", dds)
    else:
        dds = DATA_DESCRIPTOR_STR

    return TARGET_NAME + "_" + dds + "_" + "MCMC" + "_" + str(num_samples) + "num_samples" + "_" + str(num_warmup) + "num_warmup" + "_" + str(num_chains) + "num_chains" + "_" + str(thinning) + "thinning" + "_" + device


def getSMC_spec_str(TARGET_NAME, DATA_DESCRIPTOR_STR, NUM_TEMPS):

    if TARGET_NAME == "MultivariateNormalMixture":
        import target_distributions
        dds = DATA_DESCRIPTOR_STR + "_" + str(3) + "_" + str(target_distributions.MultivariateNormalMixture.FACTOR_FOR_MEAN)
        print("dds = ", dds)
    else:
        dds = DATA_DESCRIPTOR_STR

    return "smc" + "_" + TARGET_NAME + "_" + dds + "_" + str(NUM_TEMPS) + "numTemps"
  

def setGPU():
    if DATA_TYPE == "double":
        torch.set_default_dtype(torch.float64) # set double as default
    else:
        assert(DATA_TYPE == "float")
        torch.set_default_dtype(torch.float32)
    
    if torch.cuda.is_available():
        torch.cuda.set_device(get_most_freemem_gpu())
        print("USE GPU NR ", torch.cuda.current_device())
    else:
        print("USE CPU")

        # if torch.backends.mps.is_available():
        #     mps_device = torch.device("mps")
        #     x = torch.ones(1, device=mps_device)
        #     print (x)
        #     print("MPS available")
        # else:
        #     print ("MPS device not found.")
        
        # # TRY TO SPECIFY MPS HERE
        # assert(False)

    print("DATA_TYPE = ", DATA_TYPE)


def get_most_freemem_gpu():
    max_mem_ = 0
    max_id_ = 0
    i = 0 
    for g in GPUtil.getGPUs():
        if g.memoryFree > max_mem_ :
            max_mem_ = g.memoryFree
            max_id_ = i
        i += 1
    return(max_id_)

    
def filterInfValues(all_ELBO):
    if numpy.any(numpy.isinf(all_ELBO)):
        print("!! WARNING INF-VAlUE DETECTED = ", all_ELBO)
        # filter-out infinite values 
        # all_ELBO = all_ELBO[numpy.logical_not(numpy.isinf(all_ELBO))]
        all_ELBO[numpy.isinf(all_ELBO)] = numpy.nan
        print("witout inf values = ", all_ELBO)
    return all_ELBO