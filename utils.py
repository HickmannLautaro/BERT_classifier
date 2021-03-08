import glob
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
import yaml
import tensorflow as tf
import BERT_per_lvl
import BERT_per_label
from transformers import BertConfig, BertTokenizerFast
from tqdm.notebook import tqdm
import sklearn


# TODO Add docstrings

# Data analysis
def data_analysis(data_set):
    """
    Dataset analysis of occurrences and histograms of the training and test sets for
    :param data_set: dataset to analyze
    :return: display of the head of the dataset followed by a textual description fo the different categories levels. Then the same but in plot. Once for training and once for test
    """
    print("Dataset :", data_set)
    data = pd.read_csv(data_set + "/train.csv")
    data = data.rename(columns={"text": "Text", "l1": "Cat1", "l2": "Cat2", "l3": "Cat3"})

    display(data.head())

    def plot_histo(column):
        text_len = data[column].str.len()
        plt.hist(text_len, bins=text_len.max())
        plt.xlabel("Token length")
        plt.ylabel("Amount")
        plt.title("Token lenght for {}: \n Minimal: {} \n Maximal: {} \n Average: {:.2f}".format(column, text_len.min(),
                                                                                                 text_len.max(),
                                                                                                 text_len.mean()))

    def plot_histo_targets(column):
        plt.hist(data[column].str.replace(' ', '\n '), bins=len(pd.Categorical(data[column]).categories) * 2,
                 align="left");
        plt.xlabel("Label")
        plt.ylabel("Amount of appearances")
        name, count = np.unique(data[column], return_index=False, return_inverse=False, return_counts=True, axis=None)
        plt.title(
            "Amount of appearances for {}: \n Minimal: {} appears {} times \n Maximal: {} appears {} times".format(
                column, name[count.argmin()], count.min(), name[count.argmax()], count.max()))

    def get_info(column):
        name, count = np.unique(data[column], return_index=False, return_inverse=False, return_counts=True, axis=None)
        print(
            "Amount of appearances for {}: \n * unique values {}  \n * Minimal: {} appears {} times  \n * Maximal: {} appears {} times  \n * in average {:.2f} times.  \n ".format(
                column, len(count), name[count.argmin()], count.min(), name[count.argmax()], count.max(), count.mean()))

    print("Training data \nContains {} examples".format(data.shape[0]))
    get_info("Cat1")

    get_info("Cat2")

    get_info("Cat3")

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plot_histo("Text")
    plt.subplot(1, 4, 2)
    plot_histo("Cat1")
    plt.subplot(1, 4, 3)
    plot_histo("Cat2")
    plt.subplot(1, 4, 4)
    plot_histo("Cat3")

    plt.savefig("./visualizations/" + data_set + "/Data_analysis.svg", dpi=200, format="svg", facecolor="white")
    plt.show()

    test = pd.read_csv(data_set + "/test.csv")
    test = test.rename(columns={"text": "Text", "l1": "Cat1", "l2": "Cat2", "l3": "Cat3"})

    def plot_histo(column):
        text_len = test[column].str.len()
        plt.hist(text_len, bins=text_len.max())
        plt.xlabel("Token length")
        plt.ylabel("Amount")
        plt.title("Token lenght for {}: \n Minimal: {} \n Maximal: {} \n Average: {:.2f}".format(column, text_len.min(),
                                                                                                 text_len.max(),
                                                                                                 text_len.mean()))

    def plot_histo_targets(column):
        plt.hist(test[column].str.replace(' ', '\n '), bins=len(pd.Categorical(test[column]).categories) * 2,
                 align="left");
        plt.xlabel("Label")
        plt.ylabel("Amount of appearances")
        name, count = np.unique(test[column], return_index=False, return_inverse=False, return_counts=True, axis=None)
        plt.title(
            "Amount of appearances for {}: \n Minimal: {} appears {} times \n Maximal: {} appears {} times".format(
                column, name[count.argmin()], count.min(), name[count.argmax()], count.max()))

    def get_info(column):
        name, count = np.unique(test[column], return_index=False, return_inverse=False, return_counts=True, axis=None)
        print(
            "Amount of appearances for {}: \n * unique values {}  \n * Minimal: {} appears {} times  \n * Maximal: {} appears {} times  \n * in average {:.2f} times.  \n ".format(
                column, len(count), name[count.argmin()], count.min(), name[count.argmax()], count.max(), count.mean()))

    print("Test data \nContains {} examples".format(test.shape[0]))
    get_info("Cat1")

    get_info("Cat2")

    get_info("Cat3")
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plot_histo("Text")
    plt.subplot(1, 4, 2)
    plot_histo("Cat1")
    plt.subplot(1, 4, 3)
    plot_histo("Cat2")
    plt.subplot(1, 4, 4)
    plot_histo("Cat3")

    plt.savefig("./visualizations/" + data_set + "/Data_analysis_test.svg", dpi=200, format="svg", facecolor="white")
    plt.show()


def plot_sub_cat(dataset, Columns, spacer=2):
    df_empty = pd.DataFrame({'A': []})

    df_empty['Text'] = dataset[Columns[0][0]]
    if len(Columns[0]) == 2:
        df_empty['Text'] = dataset[Columns[0][1]].str.cat(df_empty['Text'], sep=". ")

    name, count = np.unique(df_empty['Text'], return_index=False, return_inverse=False, return_counts=True, axis=None)
    names_undercat_vec = []
    count_undercat_vec = []
    entries = 0
    for overcat in name:
        aux = dataset.loc[df_empty['Text'] == overcat]
        names_undercat, count_undercat = np.unique(aux[Columns[1]], return_index=False, return_inverse=False,
                                                   return_counts=True, axis=None)
        names_undercat_vec.append(names_undercat)
        names_undercat_vec.append(np.repeat(" ", spacer))

        count_undercat_vec.append(count_undercat)
        entries += len(names_undercat)
    # count_undercat_vec
    # names_undercat_vec
    plot_labels = [item for sublist in names_undercat_vec for item in sublist][:-2]
    indv_len = np.array([len(x) for x in count_undercat_vec])
    plot_pos = np.array([len(x) for x in names_undercat_vec][:-1])
    plot_pos = np.append(0, np.cumsum(plot_pos))

    y_pos = np.arange(len(plot_labels))

    ranges = [range(plot_pos[i], plot_pos[i + 1]) for i in range(0, len(plot_pos) - 1, 2)]
    for i, coun in enumerate(count_undercat_vec):
        bar_plot = plt.barh(ranges[i], coun, align='center', label=name[i])

    plt.title("Amount of appearances for under {} grouped by over categories {}:".format(Columns[1], Columns[0]))

    plt.ylabel("Label")
    plt.xscale("log")
    plt.xlabel("Amount of appearances")
    plt.yticks(y_pos, plot_labels)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left');


def plot_histo_lim(dataset, column, max_len):
    text_len = np.array([x if x <= max_len else max_len for x in dataset[column].str.len()])
    plt.hist(text_len, bins=text_len.max())
    plt.xlabel("Token length")
    plt.ylabel("Amount")
    plt.yscale("log")
    plt.title(
        "Used token lenght for {} constrained to {}: \n Minimal: {} \n Maximal: {} \n Average: {:.2f}".format(column,
                                                                                                              max_len,
                                                                                                              text_len.min(),
                                                                                                              text_len.max(),
                                                                                                              text_len.mean()))


def plot_histo_label_lim(dataset, column, cats, max_len):
    df_empty = pd.DataFrame({'A': []})
    df_empty['Text'] = dataset['Text']
    for cat in cats:
        df_empty['Text'] = dataset[cat].str.cat(df_empty['Text'], sep=". ")

    text_len = np.array([x if x <= max_len else max_len for x in df_empty['Text'].str.len()])
    plt.hist(text_len, bins=text_len.max())
    plt.xlabel("Token length")
    plt.ylabel("Amount")
    plt.yscale("log")
    plt.title(
        "Used token lenght for {}, {} as input constrained to {}: \n Minimal: {} \n Maximal: {} \n Average: {:.2f}".format(
            cats, column, max_len, text_len.min(), text_len.max(), text_len.mean()))


def plot_histo(dataset, column, max_len):
    text_len = dataset[column].str.len()
    n, _, _ = plt.hist(text_len, bins=text_len.max())
    plt.vlines(max_len, 0, n.max(), color='r')
    plt.xlabel("Token length")
    plt.ylabel("Amount")
    plt.yscale("log")
    plt.title(
        "Token lenght for {}, indicating {} as max len: \n Minimal: {} \n Maximal: {} \n Average: {:.2f}".format(column,
                                                                                                                 max_len,
                                                                                                                 text_len.min(),
                                                                                                                 text_len.max(),
                                                                                                                 text_len.mean()))


def plot_histo_label(dataset, column, cats, max_len):
    df_empty = pd.DataFrame({'A': []})
    df_empty['Text'] = dataset['Text']

    for cat in cats:
        df_empty['Text'] = dataset[cat].str.cat(df_empty['Text'], sep=". ")

    text_len = df_empty['Text'].str.len()
    n, _, _ = plt.hist(text_len, bins=text_len.max())
    plt.vlines(max_len, 0, n.max(), color='r')
    plt.xlabel("Token length")
    plt.ylabel("Amount")
    plt.yscale("log")
    plt.title(
        "Token lenght for {}, {} as input, indicating {} as max len: \n Minimal: {} \n Maximal: {} \n Average: {:.2f}".format(
            cats, column, max_len, text_len.min(), text_len.max(), text_len.mean()))


def plot_histo_targets(dataset, column):
    plt.ylabel("Label")
    plt.xscale("log")
    plt.xlabel("Amount of appearances")
    name, count = np.unique(dataset[column], return_index=False, return_inverse=False, return_counts=True, axis=None)
    plt.title(
        "Amount of appearances for {}: \n Minimal: {} appears {} times \n Maximal: {} appears {} times".format(column,
                                                                                                               name[
                                                                                                                   count.argmin()],
                                                                                                               count.min(),
                                                                                                               name[
                                                                                                                   count.argmax()],
                                                                                                               count.max()))
    y_pos = np.arange(len(name))

    bar_plot = plt.barh(y_pos, count, align='center')

    plt.yticks(y_pos, name)

    # plt.show()


def plot_histo_targets_len(dataset, column):
    plt.ylabel("Label")
    plt.xlabel("Token lenght")
    name, count = np.unique(dataset[column], return_index=False, return_inverse=False, return_counts=True, axis=None)
    lengths = np.array([len(x) for x in name])
    plt.title("Token length for {}: \n Minimal: {} is {} tokens long \n Maximal: {} is {} tokens long".format(column,
                                                                                                              name[
                                                                                                                  lengths.argmin()],
                                                                                                              lengths.min(),
                                                                                                              name[
                                                                                                                  lengths.argmax()],
                                                                                                              lengths.max()))
    y_pos = np.arange(len(name))

    bar_plot = plt.barh(y_pos, lengths, align='center')

    plt.yticks(y_pos, name)


def plot_histo_lost(dataset, column, cats, max_len):
    df_empty = pd.DataFrame({'A': []})
    df_empty['Text'] = dataset['Text']
    if cats != []:
        for cat in cats:
            df_empty['Text'] = dataset[cat].str.cat(df_empty['Text'], sep=". ")

    text_len = np.array([x - max_len for x in df_empty['Text'].str.len() if x > max_len])
    plt.hist(text_len, bins=text_len.max())
    plt.xlabel("Token length")
    plt.ylabel("Amount")
    plt.yscale("log")
    plt.title(
        "Token lenght of lost information for {}, {} as input constrained to {}: \n Minimal: {} \n Maximal: {} \n Average: {:.2f}".format(
            cats, column, max_len, text_len.min(), text_len.max(), text_len.mean()))


def data_analysis_fixed_len(data_set, max_len=100):
    print("Dataset :", data_set)
    data = pd.read_csv(data_set + "/train.csv")
    data = data.rename(columns={"text": "Text", "l1": "Cat1", "l2": "Cat2", "l3": "Cat3"})
    data = data[['Text', "Cat1", "Cat2", "Cat3"]]

    display(data.head())

    print("Training data \nContains {} examples".format(data.shape[0]))

    spec = gridspec.GridSpec(7, 3, wspace=0.5, hspace=0.6)

    fig = plt.figure(figsize=(40, 30))

    fig.add_subplot(spec[0, 0])
    plot_histo_targets(data, "Cat1")
    fig.add_subplot(spec[0, 1])
    plot_histo_targets(data, "Cat2")
    fig.add_subplot(spec[0, 2])
    plot_histo_targets(data, "Cat3")

    fig.add_subplot(spec[1, 0])
    plot_histo_targets_len(data, "Cat1")
    fig.add_subplot(spec[1, 1])
    plot_histo_targets_len(data, "Cat2")
    fig.add_subplot(spec[1, 2])
    plot_histo_targets_len(data, "Cat3")

    fig.add_subplot(spec[2, 0])
    plot_histo(data, "Text", max_len)
    fig.add_subplot(spec[2, 1])
    plot_histo_label(data, "Text", ["Cat1"], max_len)
    fig.add_subplot(spec[2, 2])
    plot_histo_label(data, "Text", ["Cat2", "Cat1"], max_len)

    fig.add_subplot(spec[3, 0])
    plot_histo_lim(data, "Text", max_len)
    fig.add_subplot(spec[3, 1])
    plot_histo_label_lim(data, "Text", ["Cat1"], max_len)
    fig.add_subplot(spec[3, 2])
    plot_histo_label_lim(data, "Text", ["Cat2", "Cat1"], max_len)

    fig.add_subplot(spec[4, 0])
    plot_histo_lost(data, "Text", [], max_len)
    fig.add_subplot(spec[4, 1])
    plot_histo_lost(data, "Text", ["Cat1"], max_len)
    fig.add_subplot(spec[4, 2])
    plot_histo_lost(data, "Text", ["Cat2", "Cat1"], max_len)

    fig.add_subplot(spec[5:, 0])
    plot_sub_cat(data, [["Cat1"], "Cat2"])
    fig.add_subplot(spec[5:, 2])
    plot_sub_cat(data, [["Cat2", "Cat1"], "Cat3"])

    plt.savefig("./visualizations/" + data_set + "/Data_analysis_complete_training.png", dpi=200, format="png",
                facecolor="white")
    plt.show()

    test = pd.read_csv(data_set + "/test.csv")
    test = test.rename(columns={"text": "Text", "l1": "Cat1", "l2": "Cat2", "l3": "Cat3"})
    test = test[['Text', "Cat1", "Cat2", "Cat3"]]

    print("Test data \nContains {} examples".format(test.shape[0]))

    fig = plt.figure(figsize=(40, 25))

    fig.add_subplot(spec[0, 0])
    plot_histo_targets(test, "Cat1")
    fig.add_subplot(spec[0, 1])
    plot_histo_targets(test, "Cat2")
    fig.add_subplot(spec[0, 2])
    plot_histo_targets(test, "Cat3")

    fig.add_subplot(spec[1, 0])
    plot_histo_targets_len(test, "Cat1")
    fig.add_subplot(spec[1, 1])
    plot_histo_targets_len(test, "Cat2")
    fig.add_subplot(spec[1, 2])
    plot_histo_targets_len(test, "Cat3")

    fig.add_subplot(spec[2, 0])
    plot_histo(test, "Text", max_len)
    fig.add_subplot(spec[2, 1])
    plot_histo_label(test, "Text", ["Cat1"], max_len)
    fig.add_subplot(spec[2, 2])
    plot_histo_label(test, "Text", ["Cat2", "Cat1"], max_len)

    fig.add_subplot(spec[3, 0])
    plot_histo_lim(test, "Text", max_len)
    fig.add_subplot(spec[3, 1])
    plot_histo_label_lim(test, "Text", ["Cat1"], max_len)
    fig.add_subplot(spec[3, 2])
    plot_histo_label_lim(test, "Text", ["Cat2", "Cat1"], max_len)

    fig.add_subplot(spec[4, 0])
    plot_histo_lost(test, "Text", [], max_len)
    fig.add_subplot(spec[4, 1])
    plot_histo_lost(test, "Text", ["Cat2"], max_len)
    fig.add_subplot(spec[4, 2])
    plot_histo_lost(test, "Text", ["Cat2", "Cat1"], max_len)

    fig.add_subplot(spec[5:, 0])
    plot_sub_cat(test, [["Cat1"], "Cat2"])
    fig.add_subplot(spec[5:, 2])
    plot_sub_cat(test, [["Cat2", "Cat1"], "Cat3"])

    plt.savefig("./visualizations/" + data_set + "/Data_analysis_complete_test.png", dpi=200, format="png",
                facecolor="white")

    plt.show()


# Result table generator

def pad(list_to_pad):
    lens = [len(a) for a in list_to_pad]
    aux = [np.pad(elem, (0, np.max(lens) - len(elem)), 'edge') for elem in list_to_pad]
    return aux


def get_plot_values(list_of_values):
    list_of_values = pad(list_of_values)
    std = np.std(list_of_values, axis=0)
    mean = np.mean(list_of_values, axis=0)
    maxim = mean + 1.96 * (std / np.sqrt(len(mean) + 1))  # np.max(f1_score_list, axis=0)
    minim = mean - 1.96 * (std / np.sqrt(len(mean) + 1))
    return mean, maxim, minim


def get_model_plot(model):
    histories = [filename for filename in glob.iglob(model + "/**/rep_and_histo.npz", recursive=True)]

    histo_list_acc = []
    histo_list_f1 = []

    for hist in histories:
        arr = np.load(hist, allow_pickle=True)
        histo = arr['hist'].item(0)
        try:
            histo_list_acc.append(np.array(histo['val_accuracy']))
            histo_list_f1.append(np.array(histo['val_f1_score']))
        except:
            histo_list_acc.append(np.array(arr['accu_list']))
            histo_list_f1.append(np.array(arr['f1_score_list']))

    plot_histo_list_acc = get_plot_values(histo_list_acc)
    plot_histo_list_f1 = get_plot_values(histo_list_f1)
    title = model[31:]
    lvl = int(title[title.find("lvl") + 3])

    return title, lvl, plot_histo_list_f1, plot_histo_list_acc


def plot_curves(models):
    fig = plt.figure(figsize=(30, 8))
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    lines = ["", '-', '--', ':']
    max_len = 0
    fig.suptitle("Validation curves while Training")
    fig.add_subplot(spec[0, 0])

    for model in models:
        title, lvl, plot_histo_list_f1, plot_histo_list_acc = get_model_plot(
            model)  # title, f1 [mean,  maxim, minim], accu [mean,  maxim, minim]
        length = plot_histo_list_f1[0].shape[0] + 1
        x = range(1, length)
        if length > max_len:
            max_len = length
        plt.plot(x, plot_histo_list_f1[0], lines[lvl], label="{} mean f1".format(title))
        plt.fill_between(x, plot_histo_list_f1[2], plot_histo_list_f1[1], alpha=0.5)

    plt.xlabel("epoch")
    plt.grid()
    plt.xticks(range(1, max_len + 1, 2))
    plt.title("mean f1 score with confidence")

    fig.add_subplot(spec[0, 1])

    for model in models:
        title, lvl, plot_histo_list_f1, plot_histo_list_acc = get_model_plot(
            model)  # title, f1 [mean,  maxim, minim], accu [mean,  maxim, minim]
        length = plot_histo_list_f1[0].shape[0] + 1
        x = range(1, length)
        plt.plot(x, plot_histo_list_acc[0], lines[lvl], label="{} mean".format(title))
        plt.fill_between(x, plot_histo_list_acc[2], plot_histo_list_acc[1], alpha=0.5)

    plt.xlabel("epoch")
    plt.legend()
    plt.grid()
    plt.xticks(range(1, max_len + 1, 2))
    plt.title("mean accuracy score with confidence")
    plt.savefig("./visualizations/trainig_curves.png", dpi=200, format="png", facecolor="white")

    plt.show()


def predict_per_label(path, input_ids, attention_mask, batch):
    imported = tf.saved_model.load(path)
    f = imported.signatures["serving_default"]
    test_pred = np.array([])
    top = input_ids.shape[0]
    for i in range(batch, top + batch, batch):
        test_pred = np.concatenate((test_pred, np.argmax(
            f(input_ids=input_ids[i - batch:i], attention_mask=attention_mask[i - batch:i])['Cat'], axis=1)))
    return test_pred


def predict(path, x, batch, test_target):
    imported = tf.saved_model.load(path)
    f = imported.signatures["serving_default"]
    test_pred = np.array([])
    top = x['input_ids'].shape[0]
    for i in range(batch, top + batch, batch):
        test_pred = np.concatenate((test_pred, np.argmax(
            f(input_ids=x['input_ids'][i - batch:i], attention_mask=x['attention_mask'][i - batch:i])['Cat'], axis=1)))
    f1_score = sklearn.metrics.f1_score(test_target, test_pred, average='macro')
    accuracy_score = sklearn.metrics.accuracy_score(test_target, test_pred)
    return f1_score, accuracy_score


def add_cats(string, lvl):
    for i in range(1, lvl):
        string += " Cat" + str(i) + ","
    string = string + " Text"
    return string


def create_results(model):
    title = model[31:]
    dataset = title[:title.find("/")]
    lvl = int(title[title.find("lvl") + 3])
    tokens = int(title[title.rfind("/") + 1:title.rfind("T")])
    epochs = title[title.rfind("T") + 2:title.rfind("e")]
    batch = int(title[title.rfind("e") + 2:title.rfind("b")])

    test_labels = None
    if title.find("flatt__") + 1:
        test_labels = None  # Depending on what shoud be tested
        train_in = "Text"
        test_in = "Text"
        return write_results(title, dataset, lvl, tokens, epochs, batch, test_labels, train_in, test_in, model)

    elif title.find("Predicted") + 1:
        train_in = add_cats("Predicted", lvl)
        test_in = train_in
        conf = "./Configs/" + dataset + "_config_lvl" + str(lvl) + "_h_p_bert-base-uncased.yaml"
        with open(conf) as f:
            arguments = yaml.load(f, Loader=yaml.FullLoader)
        test_labels = arguments["test_labels"]  # path to test labels
        return write_results(title, dataset, lvl, tokens, epochs, batch, test_labels, train_in, test_in, model)
    elif title.find("per_label") + 1:
        return write_results_per_label(title, dataset, lvl, tokens, epochs, batch, model)

    else:
        train_in = add_cats("Target", lvl)
        conf = "./Configs/" + dataset + "_config_lvl" + str(lvl) + "_h_t_bert-base-uncased.yaml"
        with open(conf) as f:
            arguments = yaml.load(f, Loader=yaml.FullLoader)
        test_labels = arguments["test_labels"]  # path to test labels

        return np.vstack((write_results(title, dataset, lvl, tokens, epochs, batch,
                                        [['Target'] + ['Cat' + str(i) for i in range(1, lvl)]], train_in, train_in,
                                        model),
                          write_results(title, dataset, lvl, tokens, epochs, batch, test_labels, train_in,
                                        add_cats("Predicted", lvl), model)))


def write_results(title, dataset, lvl, tokens, epochs, batch, test_labels, train_in, test_in, model):
    arguments = {'model_name': 'bert-base-uncased',
                 'max_length': tokens,
                 'epochs': epochs,  #
                 'batch_size': batch,
                 'data_path': dataset,
                 'lvl': lvl,
                 'test_labels': test_labels}

    model_name = arguments['model_name']
    config = BertConfig.from_pretrained(model_name)
    config.output_hidden_states = False
    data, trunest_class_names, test_target = BERT_per_lvl.get_test_data(arguments)
    x = BERT_per_lvl.get_tokenized(model_name, config, data, tokens)
    runs = [filename for filename in glob.iglob(model + "/**/model", recursive=True)]

    res_list = []
    for run in runs:
        res_list.append(predict(run, x, batch, test_target))  # f1_score, accuracy_score

    f1_mean, accu_mean = np.mean(res_list, axis=0)
    f1_std, accu_std = np.std(res_list, axis=0)
    f1_string = '{:.3f}({:.3f})'.format(f1_mean, f1_std)
    acc_string = '{:.3f}({:.3f})'.format(accu_mean, accu_std)
    aux = ['-'] * 6
    aux[(lvl - 1) * 2] = acc_string
    aux[(lvl - 1) * 2 + 1] = f1_string
    _, _, leng, _ = get_model_plot(model)
    used_ep = len(leng[0])

    table_data = ["Per_lvl", dataset, '{}({})'.format(epochs, used_ep), tokens, batch, len(runs), train_in, "Cat" + str(lvl),
                  test_in] + aux
    return table_data


def make_table(models):
    res = np.vstack([create_results(model) for model in tqdm(models)])

    df = pd.DataFrame(np.vstack(res),
                      columns=["Type", "Dataset", "Epochs", "Tokens", "Batch size", "Runs", "Train Input", "Output",
                               "Test Input", "Cat1 accuracy", "Cat1 F1 score macro", "Cat2 accuracy",
                               "Cat2 F1 score macro", "Cat3 accuracy", "Cat3 F1 score macro"])
    df = df.sort_values(by=['Dataset', 'Output', "Train Input", "Test Input"], ascending=[True, True, False, False])

    return df


def get_scores(test_pred, model, batch, x, test_target, classes, runs, dataset, lvl, tokens, epochs, train_in, test_in, prediction_only=False):
    score = []
    for run in range(1, runs + 1):
        pred = np.zeros(test_target.shape[0])
        for label_class in range(classes):
            indices_tf = [[i] for i, j in enumerate(test_pred) if j == label_class]
            indices = np.where(test_pred == label_class)[0]
            input_ids = tf.gather_nd(x['input_ids'], indices_tf)
            attention_mask = tf.gather_nd(x['attention_mask'], indices_tf)
            class_model = model + "/Run" + str(run) + "/Class" + str(label_class) + "/model"
            class_pred = predict_per_label(class_model, input_ids, attention_mask, batch).astype(int)
            mapping = np.load(model + "/Run" + str(run) + "/Class" + str(label_class) + "/tested__/rep_and_histo.npz")["test_mapping"].astype(int)
            class_pred = mapping[class_pred]
            pred[indices] = class_pred
        f1_score = sklearn.metrics.f1_score(test_target, pred, average='macro')
        accuracy_score = sklearn.metrics.accuracy_score(test_target, pred)
        score.append([f1_score, accuracy_score])
    if prediction_only:
        return pred
    f1_mean, accu_mean = np.mean(score, axis=0)
    f1_std, accu_std = np.std(score, axis=0)
    f1_string = '{:.3f}({:.3f})'.format(f1_mean, f1_std)
    acc_string = '{:.3f}({:.3f})'.format(accu_mean, accu_std)
    aux = ['-'] * 6
    aux[(lvl - 1) * 2] = acc_string
    aux[(lvl - 1) * 2 + 1] = f1_string

    table_data = ["Per_label", dataset, epochs, tokens, batch, runs, train_in, "Cat" + str(lvl), test_in] + aux
    return table_data


def write_results_per_label(title, dataset, lvl, tokens, epochs, batch, model):
    conf = "./Configs/" + dataset + "_config_lvl" + str(lvl) + "_per_label.yaml"
    with open(conf) as f:
        arguments = yaml.load(f, Loader=yaml.FullLoader)

    train_in = "Text divided per Target Cat" + str(lvl - 1)
    test_in = "Text divided per Predicted Cat" + str(lvl - 1)
    test_model = arguments["test_model_lvl1"]  # path to test labels

    model_name = arguments['model_name']
    config = BertConfig.from_pretrained(model_name)
    config.output_hidden_states = False
    data, class_names = BERT_per_label.get_upper_label_data(dataset, False, lvl)
    x = BERT_per_lvl.get_tokenized(model_name, config, data, tokens)
    runs = len([filename for filename in glob.iglob(model + "/**/Run*", recursive=True)])
    classes = class_names[lvl-2].shape[0]
    cat_num = str('Cat' + str(lvl - 1))
    cat_num_desired = str('Cat' + str(lvl))
    predicted_in = predict_per_label(test_model, x['input_ids'], x['attention_mask'], batch)
    if lvl == 3:
        second_test_model = arguments['test_model_lvl2']
        classes_for_intermediate=class_names[lvl-3].shape[0]
        predicted_in = get_scores(predicted_in, second_test_model, batch, x, np.array(data[str('Cat' + str(lvl - 1))].to_list()),classes_for_intermediate , 1, dataset, lvl - 1, tokens, epochs, train_in, train_in, True)
    test_pred = [np.array(data[cat_num].to_list()), predicted_in]
    test_target = np.array(data[cat_num_desired].to_list())
    target_in = get_scores(test_pred[0], model, batch, x, test_target, classes, runs, dataset, lvl, tokens, epochs, train_in, train_in)
    test_in = get_scores(test_pred[1], model, batch, x, test_target, classes, runs, dataset, lvl, tokens, epochs, train_in, test_in)
    return np.vstack((target_in, test_in))
