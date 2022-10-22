import logging
import re
from typing import Union
import os.path as osp
from omegaconf import DictConfig, OmegaConf
from scipy.stats import truncnorm
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import os 
import pickle
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.manifold import TSNE
#import umap

def to_clean_str(s: str) -> str:
    """Keeps only alphanumeric characters and lowers them

    Args:
        s: a string

    Returns:
        cleaned string
    """
    return re.sub("[^a-zA-Z0-9]", "", s).lower()


def display_config(cfg: DictConfig) -> None:
    """Displays the configuration"""
    logger = logging.getLogger()
    logger.info("Configuration:\n")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 40 + "\n")


def flatten(d: Union[dict, list], parent_key: str = "", sep: str = ".") -> dict:
    """Flattens a dictionary or list into a flat dictionary

    Args:
        d: dictionary or list to flatten
        parent_key: key of parent dictionary
        sep: separator between key and child key

    Returns:
        flattened dictionary

    """
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            items.extend(flatten(v, new_key, sep=sep).items())
    elif isinstance(d, list):
        for i, elem in enumerate(d):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.extend(flatten(elem, new_key, sep).items())
    else:
        items.append((parent_key, d))
    return dict(items)


def truncated_normal(size, threshold=1):
    """Samples values from truncated normal distribution centered at 0

    Args:
        size: shape or amount of samples
        threshold: cut-off value for distribution

    Returns:
        numpy array of given size

    """
    return truncnorm.rvs(-threshold, threshold, size=size)


def plot_features(features, labels, num_classes, epoch, prefix,save_dir):
    """Plot features on 2D plane.

    Args:
        features: (num_instances, num_features).
        labels: (num_instances). 
    """
    plt.rcParams['font.family']='Times New Roman'#'SimHei'为黑体
    plt.rcParams['axes.unicode_minus'] = False #显示负号
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    #plt.grid(linestyle='--',linewidth=0.5)  # 网格
    plt.figure(num=0,figsize=(8.5,6),dpi=600) # 控制图片大小和质量
    sns.set_context("paper")
    
    if features.shape[1] >= 5 and features.shape[0]<20000:
        # reducer = umap.UMAP(random_state=42)
        # embedding = reducer.fit_transform(features)
        # print(embedding.shape)
        print("n_components",features.shape,features)
        tsne = TSNE(n_components=2,init='pca', random_state=500)
        tsne.fit_transform(features)
        features = tsne.embedding_
        print(tsne.embedding_)
        #print("n_components",features)

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(num_classes):
        plt.scatter(
            features[labels==label_idx, 0],
            features[labels==label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    #plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right',markerscale=3)
    #
    plt.xlabel(r'Activation of the 1st neuron',fontsize=15)  # 设置x轴标签
    plt.ylabel(r'Activation of the 2nd neuron',fontsize=15)  # 设置x轴标签
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    dirname = osp.join(save_dir, prefix)
    if not osp.exists(dirname):
        os.mkdir(dirname)
    save_name = osp.join(dirname, 'epoch_' + str(epoch+1) + '.pdf')
    plt.savefig(save_name, bbox_inches='tight',dpi=1000)
    plt.close()




def plot_hist(data_clean_pred,data_noise_pred,epoch,prefix,save_dir):
    plt.rcParams['font.family']='Times New Roman'#'SimHei'为黑体
    plt.rcParams['axes.unicode_minus'] = False #显示负号
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.grid(linestyle='--',linewidth=0.5)  # 网格
    plt.figure(num=0,figsize=(8.5,6),dpi=600) # 控制图片大小和质量
    sns.set_context("paper")

    #fig,ax=plt.subplots(1)
    plt.cla()
    plt.hist(data_clean_pred, bins=20, range=(0., 1.), edgecolor='black', alpha=0.5, label='clean')
    plt.hist(data_noise_pred, bins=20, range=(0., 1.), edgecolor='black', alpha=0.5, label='noisy')
    plt.xlabel('probability',fontsize=20)
    plt.ylabel('number of data',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.grid()
    #file_name = os.path.join(save_dir,'histogram_sep_epoch%03d'%(epoch).jpg)
    dirname = osp.join(save_dir, prefix)
    if not osp.exists(dirname):
        os.mkdir(dirname)
    save_name = osp.join(dirname, 'epoch_' + str(epoch+1) + '.pdf')
    plt.savefig(save_name, bbox_inches='tight',dpi=1000)
    plt.close()


def plot_loss(data_clean_grad,data_noise_grad,epoch,total_epoch,prefix,save_dir):

    #total_epoch = args['trainer']['epochs']
    plt.rcParams['font.family']='Times New Roman'#'SimHei'为黑体
    plt.rcParams['axes.unicode_minus'] = False #显示负号
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.grid(linestyle='--',linewidth=0.5)  # 网格
    plt.figure(num=0,figsize=(8.5,6),dpi=600) # 控制图片大小和质量
    sns.set_context("paper")

    #fig,ax=plt.subplots(2)

    y1 = data_clean_grad
    y2 = data_noise_grad

    plt.cla()
    #ax.set_title("Loss_analysis")
    plt.xlabel("Epoch",fontsize=20)
    plt.ylabel(r"$|\partial{\mathcal{L}}/\partial{p_y}|$",fontsize=20)
    plt.xlim(0,total_epoch)
    #ax.set_ylim(-10,1)
    plt.grid()
    plt.plot(y1,label='True',linewidth=2)
    plt.plot(y2,label='Wrong',linewidth=2)
    plt.legend(loc='best',fontsize=20)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    #file_name = os.path.join(save_dir,'histogram_sep_epoch%03d'%(epoch).jpg)
    dirname = osp.join(save_dir, prefix)
    if not osp.exists(dirname):
        os.mkdir(dirname)
    save_name = osp.join(dirname, 'epoch_' + str(epoch+1) + '.pdf')
    plt.savefig(save_name, bbox_inches='tight',dpi=1000)
    plt.close()
    
    #
def plot_dynamic_acc(data_clean_grad,data_noise_grad,epoch,args):
    
    save_dir = args.save_dir

    total_epoch = args['trainer']['epochs']

    fig,ax=plt.subplots()

    y1 = data_clean_grad
    y2 = data_noise_grad

    ax.cla()
    ax.set_title("Grad_analysis")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Grad")
    ax.set_xlim(0,total_epoch)
    #ax.set_ylim(-10,1)
    ax.grid()
    ax.plot(y1,label='clean_grad')
    ax.plot(y2,label='noise_grad')
    ax.legend(loc='best')

    #file_name = os.path.join(save_dir,'histogram_sep_epoch%03d'%(epoch).jpg)
    plt.savefig(str(save_dir)+'/Grad_analysis.pdf')



def plot_confusion_matrix(y_true, y_pred, dataset_name,
                          normalize=False,
                          title=None,
                          prefix=None,
                          save_dir=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if dataset_name == 'mnist':
        classes = np.arange(10)
        xylabels = classes
    elif dataset_name == 'cifar10':
        classes = np.arange(10)
        xylabels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    elif dataset == 'cifar100':
        classes = np.arange(100)
        xylabels = classes
    else:
        assert 1 == 0
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')


    print("confusion_matrix",cm)



    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=xylabels, yticklabels=xylabels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)


    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    dirname = osp.join(save_dir, prefix)
    if not osp.exists(dirname):
        os.mkdir(dirname)
    save_name = osp.join(dirname, 'confusion_matrix' + '.pdf')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

    return ax
