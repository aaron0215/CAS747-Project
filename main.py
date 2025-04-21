from dataclasses import dataclass
from attack.nifa import NIFA
from models.bayesianModel import *
from models.victimModels import *
from utils.dataLoader import load_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"running on device: ", device)

models = ['GCN', 'SGC', 'APPNP', 'GraphSAGE']
# models = ['GCN']
# datasets = ['pokec_z', 'pokec_n', 'dblp']
datasets = ['dblp']
final_data = dict.fromkeys(datasets)
init_accuracy = dict.fromkeys(models)
init_sp = dict.fromkeys(models)
init_eo = dict.fromkeys(models)
poisoned_accuracy = dict.fromkeys(models)
poisoned_sp = dict.fromkeys(models)
poisoned_eo = dict.fromkeys(models)

# hyper parameters
@dataclass
class pokec_z_param:
    learning_rate = 0.001
    dropout_ratio = 0
    epochs = 1000
    n = 102  # injected node budget
    d = 50  # degree of injected node
    patience = 50  # early stopping patience
    theta = 0.5  # bernoulli parameter for BayesianGNN
    max_iter = 20  # NIFA attack parameter
    max_steps = 50  # NIFA attack parameter
    T = 20  # sampling times of BayesianGNN
    k = 0.5  # top k% highest-uncertainty node are attacked
    # parameter for overall loss calculation in attack
    alpha = 0.01  # weight of L_cf
    beta = 4  # weight of L_sp + L_eo

@dataclass
class pokec_n_param:
    learning_rate = 0.001
    dropout_ratio = 0
    epochs = 1000
    n = 87  # injected node budget
    d = 50  # degree of injected node
    patience = 50  # early stopping patience
    theta = 0.5  # bernoulli parameter for BayesianGNN
    max_iter = 20  # NIFA attack parameter
    max_steps = 50  # NIFA attack parameter
    T = 20  # sampling times of BayesianGNN
    k = 0.5  # top k% highest-uncertainty node are attacked
    # parameter for overall loss calculation in attack
    alpha = 0.01  # weight of L_cf
    beta = 4  # weight of L_sp + L_eo

@dataclass
class dblp_param:
    learning_rate = 0.001
    dropout_ratio = 0
    epochs = 1000
    n = 32  # injected node budget
    d = 24  # degree of injected node
    patience = 50  # early stopping patience
    theta = 0.5  # bernoulli parameter for BayesianGNN
    max_iter = 10  # NIFA attack parameter
    max_steps = 50  # NIFA attack parameter
    T = 20  # sampling times of BayesianGNN
    k = 0.5  # top k% highest-uncertainty node are attacked
    # parameter for overall loss calculation in attack
    alpha = 0.1  # weight of L_cf
    beta = 8  # weight of L_sp + L_eo

params = {
    "pokec_z": pokec_z_param(),
    "pokec_n": pokec_n_param(),
    "dblp": dblp_param()
}

# control multiple runs
iteration_times = 2

for dataset in datasets:
    if final_data[dataset] is None:
        final_data[dataset] = []
    for run in range(iteration_times):
        # load data
        path = './data/{dataset}.bin'.format(dataset=dataset)
        g, index_split = load_data(path)
        g = g.to(device)
        feature = g.ndata['feature'].shape[1]
        hid_dimension = 128
        num_classes = max(g.ndata['label']).item() + 1
        labels = g.ndata['label']
        result_path = f"./output/{dataset}_best.bin"
        param = params[dataset]
        for model in models:
            if init_accuracy[model] is None:
                init_accuracy[model] = []
            if init_sp[model] is None:
                init_sp[model] = []
            if init_eo[model] is None:
                init_eo[model] = []
            if poisoned_accuracy[model] is None:
                poisoned_accuracy[model] = []
            if poisoned_sp[model] is None:
                poisoned_sp[model] = []
            if poisoned_eo[model] is None:
                poisoned_eo[model] = []
            print(f"+++++++++++++++ {model} model pre-attack training starts +++++++++++++++++")
            victim_model = VictimModels(feature, hid_dimension, num_classes, device, param.dropout_ratio, name=model,
                                        training=True)
            victim_model.train_model(g, index_split, param.epochs, param.learning_rate, result_path)
            acc, sp, eo, delta_sp, delta_eo = victim_model.test_model(g, index_split, result_path)
            init_accuracy[model].append(acc)
            init_sp[model].append(delta_sp)
            init_eo[model].append(delta_eo)
            print(f"+++++++++++++++ {model} model pre-attack training finished +++++++++++++++++")

            print(f"=============== start attack on {model} model ===================")
            nifa_instance = NIFA(g, feature, hid_dimension, num_classes, device, param.T, param.theta, node_budget=param.n, edge_budget=param.d)
            g_attack, uncertainty = nifa_instance.attack(g, index_split, param.learning_rate, param.k,
                                                param.max_iter, param.max_steps, param.alpha, param.beta)  # uncertainty shape: [n_nodes]
            print(f"=============== finish attack on {model} model ===================")

            torch.save( [g], f"./output/{dataset}_injected.bin")

            print(f"--------------- {model} model after-attack training starts ------------------")
            victim_model_retrain = VictimModels(feature, hid_dimension, num_classes, device, param.dropout_ratio, name=model,
                                        training=False)
            victim_model_retrain.train_with_removal(g_attack, index_split, param.epochs, param.learning_rate, result_path, uncertainty)
            acc_atk, sp_atk, eo_atk, delta_sp_atk, delta_eo_atk = victim_model_retrain.test_model(g_attack, index_split, result_path)
            poisoned_accuracy[model].append(acc_atk)
            poisoned_sp[model].append(delta_sp_atk)
            poisoned_eo[model].append(delta_eo_atk)
            print(f"--------------- {model} model after-attack training finished ------------------")

    final_data[dataset].append(init_accuracy)
    final_data[dataset].append(init_sp)
    final_data[dataset].append(init_eo)
    final_data[dataset].append(poisoned_accuracy)
    final_data[dataset].append(poisoned_sp)
    final_data[dataset].append(poisoned_eo)

# final result output
str_formatter = lambda x, y: "{:.2f}±{:.2f}".format(np.mean(x[y])*100, np.std(x[y])*100)
for dataset in datasets:
    print(f"\033[44;33m{dataset}\033[0m")
    # print(final_data)
    # print(final_data[dataset])
    data = final_data[dataset]
    # print(data)
    for model in models:
        print(f"\033[95m{model}\033[0m")
        print(f"initial accuracy: {str_formatter(data[0],model)}")
        print(f"initial sp: {str_formatter(data[1],model)}")
        print(f"initial eo: {str_formatter(data[2],model)}")
        print(f"after-poisoned accuracy: {str_formatter(data[3],model)}")
        print(f"after-poisoned sp: {str_formatter(data[4],model)}")
        print(f"after-poisoned eo: {str_formatter(data[5],model)}")



